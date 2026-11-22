"""Split a sampler shared by comparison and ordinary sampling in the bridge's named SPIR-V profile.

SPIRV-Cross declares one HLSL sampler per SPIR-V sampler variable. When one variable feeds both
OpImageSampleDrefExplicitLod and OpImageSampleExplicitLod, it emits a SamplerComparisonState and
passes it to an ordinary SampleLevel call, which DXC rejects. This pass keeps the comparison
sampler at its original binding and moves only the ordinary uses to a new sampler variable at a
binding the caller supplies. The caller (the host) owns that binding and must bind an ordinary
sampler there.

The pass first looks for a standalone UniformConstant sampler, loaded directly, whose sampled
images feed both kinds of sampling. Modules without one, including indexed sampler tables and
unrelated sampling, are returned byte-for-byte unchanged. For the sampler being split, the profile
is limited to direct, non-array float32 Texture2D sampling in a single compute entry point, at a
constant LOD of zero, with the sampler listed in the entry-point interface. Any other use of that
sampler is rejected with a specific diagnostic.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import sys

PREFIX = "%x2w_plain_sampler"
ZERO = {"0", "0.0", "0x0p+0", "-0", "-0.0", "-0x0p+0"}
# Debug and annotation instructions that name an ID without using its value.
ANNOTATIONS = {"OpName", "OpMemberName", "OpEntryPoint", "OpExecutionMode",
               "OpDecorate", "OpMemberDecorate", "OpDecorateId", "OpDecorateString", "OpMemberDecorateString",
               "OpDecorateStringGOOGLE", "OpMemberDecorateStringGOOGLE"}


def _tokens(line):
    lexer = shlex.shlex(line, posix=False)
    lexer.commenters = ";"
    lexer.whitespace_split = True
    return list(lexer)


def _standalone_sampler(definitions, loaded):
    """Return the variable behind a direct OpLoad of a standalone UniformConstant sampler, else None."""
    load = definitions.get(loaded, [])
    if len(load) != 3 or load[0] != "OpLoad":
        return None
    variable = definitions.get(load[2], [])
    if len(variable) != 3 or variable[0] != "OpVariable" or variable[2] != "UniformConstant":
        return None
    pointer = definitions.get(variable[1], [])
    if (len(pointer) != 3 or pointer[:2] != ["OpTypePointer", "UniformConstant"] or pointer[2] != load[1]
            or definitions.get(pointer[2]) != ["OpTypeSampler"]):
        return None
    return load[2]


def split_comparison_sampler(assembly, ordinary_binding):
    """Return (assembly, mapping). mapping is None when no candidate sampler is found.

    A candidate is a directly loaded standalone sampler whose sampled images feed both comparison and
    ordinary sampling. None does not mean the module is free of mixed samplers: one reached through
    another path (a copied pointer or a load with memory operands, for example) is not a candidate
    and is returned unchanged.
    """
    if ordinary_binding < 0:
        raise ValueError("Ordinary sampler binding must be nonnegative")
    lines = assembly.splitlines()
    parsed = [_tokens(line) for line in lines]
    definitions = {words[0]: words[2:] for words in parsed if len(words) >= 3 and words[1] == "="}
    defined_at = {words[0]: i for i, words in enumerate(parsed) if len(words) >= 3 and words[1] == "="}

    # Every non-annotation instruction that references an ID, keyed by that ID.
    uses = defaultdict(list)
    for words in parsed:
        args = words[2:] if len(words) >= 3 and words[1] == "=" else words
        if args and args[0] not in ANNOTATIONS:
            for operand in args[1:]:
                if operand.startswith("%"):
                    uses[operand].append((words[0] if args is not words else None, args))

    # 1. Find candidates without judging anything outside the standalone profile: a sampled image
    # whose sampler is a direct OpLoad of a standalone UniformConstant OpTypeSampler variable.
    kinds_by_sampler = defaultdict(set)
    for image, definition in definitions.items():
        if definition[:1] != ["OpSampledImage"] or len(definition) != 4:
            continue
        sampler = _standalone_sampler(definitions, definition[3])
        if sampler is None:
            continue
        for _, use in uses[image]:
            if use[0].startswith("OpImage") and use[0] != "OpImage":
                kinds_by_sampler[sampler].add("compare" if "Dref" in use[0] else "ordinary")
    mixed = [sampler for sampler, kinds in kinds_by_sampler.items() if len(kinds) == 2]
    if not mixed:
        return assembly, None
    if len(mixed) != 1:
        raise ValueError("Sampler split supports one sampler with mixed uses per module")
    sampler = mixed[0]

    # 2. Apply the narrow profile to the sampler being split, and only to it.
    entries = [i for i, words in enumerate(parsed) if words[:1] == ["OpEntryPoint"]]
    if len(entries) != 1 or parsed[entries[0]][1] != "GLCompute":
        raise ValueError("Sampler split requires a single GLCompute entry point")
    entry = entries[0]
    images = []
    for _, use in uses[sampler]:
        if use[0] != "OpLoad":
            raise ValueError(f"{sampler}: {use[0]} uses the sampler; only direct loads are supported")
    for load in (result for result, use in uses[sampler]):
        for image, use in uses[load]:
            if use[0] != "OpSampledImage" or use[-1] != load:
                raise ValueError(f"{sampler}: loaded sampler feeds {use[0]}; only OpSampledImage is supported")
            images.append(image)
    ordinary_images = []
    for image in images:
        sampled = definitions.get(definitions[image][1], [])
        texture = definitions.get(sampled[1], []) if sampled[:1] == ["OpTypeSampledImage"] else []
        # OpTypeImage <sampled type> 2D <depth> <arrayed> <ms> <sampled> <format>
        if (len(texture) != 8 or texture[0] != "OpTypeImage"
                or definitions.get(texture[1]) != ["OpTypeFloat", "32"]
                or texture[2] != "2D" or texture[4:] != ["0", "0", "1", "Unknown"]):
            raise ValueError(f"{image}: only non-array float32 Texture2D sampling is supported")
        kinds = set()
        for _, use in uses[image]:
            ordinary = use[0] == "OpImageSampleExplicitLod" and len(use) == 6
            compare = use[0] == "OpImageSampleDrefExplicitLod" and len(use) == 7
            if not (ordinary or compare) or use[2] != image or use[-2] != "Lod":
                raise ValueError(f"{image}: only explicit-LOD ordinary and depth-comparison sampling is supported")
            lod = definitions.get(use[-1], [])
            if len(lod) != 3 or lod[:2] != ["OpConstant", texture[1]] or lod[2] not in ZERO:
                raise ValueError(f"{image}: {use[0]} requires a constant LOD of zero")
            kinds.add("ordinary" if ordinary else "compare")
        if len(kinds) > 1:
            raise ValueError(f"{image}: one sampled image is used for both sampling kinds")
        if kinds == {"ordinary"}:
            ordinary_images.append(image)

    bindings = {}
    descriptor_sets = {}
    for i, words in enumerate(parsed):
        if len(words) == 4 and words[0] == "OpDecorate" and words[2] in ("Binding", "DescriptorSet"):
            try:
                value = int(words[3], 0)
            except ValueError:
                raise ValueError(f"{words[1]}: unreadable {words[2]} decoration") from None
            if words[2] == "Binding":
                bindings[words[1]] = (i, value)
            else:
                descriptor_sets[words[1]] = value
    if sampler not in parsed[entry][4:]:
        raise ValueError(f"{sampler}: sampler must be listed in the entry-point interface")
    if sampler not in bindings or sampler not in descriptor_sets:
        raise ValueError(f"{sampler}: sampler needs a DescriptorSet and Binding decoration")
    binding_line, comparison_binding = bindings[sampler]
    descriptor_set = descriptor_sets[sampler]
    for variable, (_, slot) in bindings.items():
        # A Binding without a DescriptorSet decoration is counted as set 0, so it cannot hide.
        if descriptor_sets.get(variable, 0) == descriptor_set and slot == ordinary_binding:
            raise ValueError(f"Ordinary sampler binding {ordinary_binding} is occupied in set "
                             f"{descriptor_set} by {variable}")
    if any(word.startswith(PREFIX) for words in parsed for word in words):
        raise ValueError("Reserved sampler split ID is already in use")

    lines[entry] = " ".join([*parsed[entry], PREFIX])
    lines[binding_line] += (f"\nOpDecorate {PREFIX} DescriptorSet {descriptor_set}"
                            f"\nOpDecorate {PREFIX} Binding {ordinary_binding}")
    lines[defined_at[sampler]] += f"\n{PREFIX} = OpVariable {definitions[sampler][1]} UniformConstant"
    sampler_type = definitions[definitions[sampler][1]][2]
    for index, image in enumerate(ordinary_images):
        i = defined_at[image]
        load = f"{PREFIX}_{index}"
        lines[i] = f"{load} = OpLoad {sampler_type} {PREFIX}\n" + " ".join([*parsed[i][:-1], load])
    mapping = {"descriptor_set": descriptor_set, "comparison_binding": comparison_binding,
               "ordinary_binding": ordinary_binding}
    return "\n".join(lines) + "\n", mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="SPIR-V assembly")
    parser.add_argument("output", type=Path, help="lowered SPIR-V assembly")
    parser.add_argument("--ordinary-binding", type=int, required=True,
                        help="free binding, in the comparison sampler's descriptor set, for the ordinary sampler")
    args = parser.parse_args()
    try:
        assembly, mapping = split_comparison_sampler(args.input.read_text(encoding="utf-8"),
                                                     args.ordinary_binding)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.output.write_text(assembly, encoding="utf-8", newline="\n")
    if mapping:
        print(" ".join(f"{key}={value}" for key, value in mapping.items()))
    else:
        print("unchanged: no candidate sampler found (a directly loaded standalone sampler "
              "with comparison and ordinary uses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
