"""Split the #670 shared comparison/ordinary sampler, compile it and check the GPU readback.

Usage: compare_sampler.py <spirv-as> <spirv-val> <spirv-cross> [<dxc> <offscreen_check> hardware|warp]

Without the last three arguments only the SPIR-V and HLSL stages and the rejection controls run.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from spirv_sampler_split import split_comparison_sampler
from spirv_sampler_views import split_sampler_views

ORDINARY_BINDING = 5
SHARED = Path(__file__).with_name("shared_sampler.spvasm").read_text(encoding="utf-8")
# 0.25 <= 0.5 passes, 0.75 <= 0.5 fails, ordinary sample of 0.375; the other 13 words are guards.
EXPECTED = [0x3f800000, 0x00000000, 0x3ec00000] + [0xc5c5c5c5] * 13


def substitute(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def rejected(assembly, binding, reason):
    try:
        split_comparison_sampler(assembly, binding)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f"accepted invalid input: {reason}")


def controls(directory):
    lowered, mapping = split_comparison_sampler(SHARED, ORDINARY_BINDING)
    assert mapping == {"descriptor_set": 0, "comparison_binding": 2, "ordinary_binding": ORDINARY_BINDING}, mapping
    # Modules without a mixed standalone sampler come back byte-for-byte, whatever else they contain.
    separated = substitute(SHARED, "%41 = OpLoad %type_sampler %cmpSampler", "%41 = OpLoad %type_sampler %plainSampler")
    separated_lod = substitute(separated, "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
                               "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0_25")
    table = Path(__file__).with_name("sampler.spvasm").read_text(encoding="utf-8")  # #676 indexed table
    for name, module in (("already separated", separated), ("unrelated nonzero LOD", separated_lod),
                         ("indexed sampler table", table), ("split sampler views", split_sampler_views(table))):
        assert split_comparison_sampler(module, ORDINARY_BINDING) == (module, None), name

    # An unrelated ordinary sample at nonzero LOD on another sampler does not block the split.
    unrelated = substitute(SHARED, "%44 = OpCompositeExtract %float %43 0",
                           "%44 = OpCompositeExtract %float %43 0\n"
                           "%u1 = OpLoad %type_sampler %plainSampler\n"
                           "%u2 = OpSampledImage %type_sampled_image %40 %u1\n"
                           "%u3 = OpImageSampleExplicitLod %v4float %u2 %15 Lod %float_0_25")
    split, mapping = split_comparison_sampler(unrelated, ORDINARY_BINDING)
    assert mapping["ordinary_binding"] == ORDINARY_BINDING
    assert "%u2 = OpSampledImage %type_sampled_image %40 %u1" in split

    # String and ID decorations name the sampler without using it, so they must not block the split.
    decorated = substitute(SHARED, "OpDecorate %cmpSampler Binding 2",
                           'OpDecorate %cmpSampler Binding 2\nOpDecorateString %cmpSampler UserSemantic "S"')
    decorated_split, mapping = split_comparison_sampler(decorated, ORDINARY_BINDING)
    assert mapping["ordinary_binding"] == ORDINARY_BINDING
    assert 'OpDecorateString %cmpSampler UserSemantic "S"' in decorated_split

    # Unsupported uses of the sampler being split are still rejected.
    for binding in (2, 3, 4):
        rejected(SHARED, binding, f"binding {binding} is occupied in set 0")
    # A binding declared without a DescriptorSet decoration still occupies set 0.
    rejected(substitute(SHARED, "OpDecorate %plainSampler DescriptorSet 0\n", ""), 3,
             "binding 3 is occupied in set 0 by %plainSampler")
    # The ordinary sample may not reuse the comparison's sampled image.
    rejected(substitute(SHARED, "%43 = OpImageSampleExplicitLod %v4float %42 ",
                        "%43 = OpImageSampleExplicitLod %v4float %34 "),
             ORDINARY_BINDING, "one sampled image is used for both sampling kinds")
    rejected(substitute(SHARED, "%float_0_75 Lod %float_0", "%float_0_75 Lod %float_0_25"),
             ORDINARY_BINDING, "requires a constant LOD of zero")
    rejected(substitute(SHARED, "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0",
                        "%43 = OpImageSampleExplicitLod %v4float %42 %15 Lod %float_0_25"),
             ORDINARY_BINDING, "requires a constant LOD of zero")
    rejected(substitute(SHARED, "%33 = OpLoad %type_sampler %cmpSampler",
                        "%copy = OpCopyObject %_ptr_UniformConstant_type_sampler %cmpSampler\n"
                        "%33 = OpLoad %type_sampler %cmpSampler"),
             ORDINARY_BINDING, "only direct loads are supported")

    tool = [sys.executable, "-B", str(ROOT / "tools" / "spirv_sampler_split.py")]
    source = directory / "shared.spvasm"
    source.write_text(SHARED, encoding="utf-8")
    target = directory / "lowered.spvasm"
    run = subprocess.run([*tool, source, target, "--ordinary-binding", str(ORDINARY_BINDING)],
                         capture_output=True, text=True, check=True)
    assert run.stdout.split() == ["descriptor_set=0", "comparison_binding=2", f"ordinary_binding={ORDINARY_BINDING}"]
    assert target.read_text(encoding="utf-8") == lowered
    occupied = subprocess.run([*tool, source, directory / "occupied.spvasm", "--ordinary-binding", "3"],
                              capture_output=True, text=True)
    assert occupied.returncode == 1 and "occupied" in occupied.stderr, occupied
    # Without a candidate, the CLI reports what it looked for rather than claiming there is no mixed use.
    separate = directory / "separate.spvasm"
    separate.write_text(separated, encoding="utf-8")
    none = subprocess.run([*tool, separate, directory / "unchanged.spvasm", "--ordinary-binding", "5"],
                          capture_output=True, text=True, check=True)
    assert none.stdout.startswith("unchanged: no candidate sampler found"), none.stdout
    assert (directory / "unchanged.spvasm").read_text(encoding="utf-8") == separated
    assert not (directory / "occupied.spvasm").exists()
    return lowered, decorated_split


def emit(directory, assembler, validator, cross):
    lowered, decorated_split = controls(directory)
    outputs = {}
    for name, assembly in (("shared", SHARED), ("lowered", lowered), ("decorated", decorated_split)):
        text = directory / f"{name}.spvasm"
        text.write_text(assembly, encoding="utf-8")
        module = directory / f"{name}.spv"
        subprocess.run([assembler, "--target-env", "vulkan1.2", text, "-o", module], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
        hlsl = directory / f"{name}.hlsl"
        subprocess.run([cross, module, "--hlsl", "--shader-model", "66", "--output", hlsl], check=True)
        outputs[name] = hlsl
    source = outputs["lowered"].read_text(encoding="utf-8")
    # The host's static samplers live in space 0, so the declarations must name it.
    comparison = re.search(r"SamplerComparisonState\s+(\w+)\s*:\s*register\(s2,\s*space0\)", source)
    ordinary = re.search(rf"SamplerState\s+(\w+)\s*:\s*register\(s{ORDINARY_BINDING},\s*space0\)", source)
    assert comparison and ordinary, source
    assert source.count(f"SampleCmpLevelZero({comparison[1]},") == 2, source
    assert source.count(f"SampleLevel({ordinary[1]},") == 1, source
    print(f"shared and lowered SPIR-V validated; comparison s2, ordinary s{ORDINARY_BINDING}; controls passed",
          flush=True)
    return outputs


def compile_dxil(directory, dxc, outputs):
    # The unsplit module reproduces the original DXC rejection.
    shared = subprocess.run([dxc, "-T", "cs_6_6", "-E", "main", "-Fo", directory / "shared.dxil", outputs["shared"]],
                            capture_output=True, text=True)
    if shared.returncode == 0 or "no matching member function for call to 'SampleLevel'" not in shared.stderr:
        raise AssertionError(f"Expected the shared-sampler DXC diagnostic\n{shared.stdout}\n{shared.stderr}")
    print("shared-sampler HLSL: DXC rejected SampleLevel as expected", flush=True)
    dxil = directory / "lowered.dxil"
    subprocess.run([dxc, "-T", "cs_6_6", "-E", "main", "-Fo", dxil, outputs["lowered"]], check=True)
    return dxil


def with_words(**words):
    reference = EXPECTED.copy()
    for index, value in words.items():
        reference[int(index[1:])] = value
    return reference


# (name, descriptor variant, reference, mismatching words reported on stderr)
DRAWS = [
    ("baseline", "baseline", EXPECTED, []),
    # Wrong reference: claim the failing comparison passed.
    ("wrong reference", "baseline", with_words(w1=0x3f800000), ["word=1 expected=3f800000 actual=00000000"]),
    # s2 switched to GREATER: only the two comparison words change, against the baseline reference.
    ("comparison descriptor vs baseline", "comparison-greater", EXPECTED,
     ["word=0 expected=3f800000 actual=00000000", "word=1 expected=00000000 actual=3f800000"]),
    ("comparison descriptor", "comparison-greater", with_words(w0=0x00000000, w1=0x3f800000), []),
    # Ordinary sampler clamped to MinLOD 1: only the ordinary word changes, to t1's 1x1 mip (0.625).
    ("ordinary descriptor vs baseline", "ordinary-min-lod1", EXPECTED,
     ["word=2 expected=3ec00000 actual=3f200000"]),
    ("ordinary descriptor", "ordinary-min-lod1", with_words(w2=0x3f200000), []),
]


def draw(directory, dxil, renderer, adapter):
    reference = directory / "reference.bin"
    for name, variant, expected, mismatches in DRAWS:
        reference.write_bytes(struct.pack("<16I", *expected))
        run = subprocess.run([renderer, adapter, "--spirv-compare-sampler", dxil, str(ORDINARY_BINDING), reference,
                              variant], capture_output=True, text=True)
        print(f"[{name}]", run.stdout, run.stderr, flush=True)
        if run.returncode == 77 and name == "baseline":
            return 77
        reported = [line for line in run.stderr.splitlines() if line.startswith("word=")]
        if (run.returncode != int(bool(mismatches)) or reported != mismatches
                or f"descriptors={variant} " not in run.stdout
                or f"compared_words=16 mismatches={len(mismatches)}" not in run.stdout):
            raise RuntimeError(f"Unexpected {name} readback")
    print("wrong reference fails only at word 1; s2 changes only words 0-1; "
          f"s{ORDINARY_BINDING} changes only word 2", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (4, 7):
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        outputs = emit(directory, *sys.argv[1:4])
        if len(sys.argv) == 7:
            dxil = compile_dxil(directory, sys.argv[4], outputs)
            sys.exit(draw(directory, dxil, *sys.argv[5:]))
