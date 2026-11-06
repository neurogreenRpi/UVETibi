from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tensorflow.lite.python import schema_py_generated as schema


SOURCE = Path(
    "/mnt/a/Camion/runs/detect/runs/detect/"
    "yolo_camions_edgetpu_416/weights/best_saved_model/"
    "best_full_integer_quant.tflite"
)

DESTINATION = SOURCE.with_name(
    "best_full_integer_quant_resize_v2.tflite"
)


def create_interpreter(*, path=None, content=None):
    kwargs = {"num_threads": 2}

    if path is not None:
        kwargs["model_path"] = str(path)
    else:
        kwargs["model_content"] = content

    interpreter = Interpreter(**kwargs)
    interpreter.allocate_tensors()
    return interpreter


def compare_models(original, modified):
    inputs_a = original.get_input_details()
    inputs_b = modified.get_input_details()
    outputs_a = original.get_output_details()
    outputs_b = modified.get_output_details()

    if len(inputs_a) != 1 or len(inputs_b) != 1:
        raise RuntimeError("Le test attend un modèle à une seule entrée.")

    a = inputs_a[0]
    b = inputs_b[0]

    if not np.array_equal(a["shape"], b["shape"]):
        raise RuntimeError("La forme d'entrée a changé.")

    if a["dtype"] != np.int8 or b["dtype"] != np.int8:
        raise RuntimeError("Le test attend des entrées INT8.")

    if a["quantization"] != b["quantization"]:
        raise RuntimeError("La quantification d'entrée a changé.")

    if len(outputs_a) != len(outputs_b):
        raise RuntimeError("Le nombre de sorties a changé.")

    shape = tuple(int(n) for n in a["shape"])
    rng = np.random.default_rng(42)

    tests = (
        np.full(shape, -128, dtype=np.int8),
        np.zeros(shape, dtype=np.int8),
        rng.integers(-128, 128, size=shape, dtype=np.int8),
    )

    for test_index, image in enumerate(tests, start=1):
        original.set_tensor(a["index"], image)
        modified.set_tensor(b["index"], image)

        original.invoke()
        modified.invoke()

        for output_index, (oa, ob) in enumerate(
            zip(outputs_a, outputs_b)
        ):
            if oa["quantization"] != ob["quantization"]:
                raise RuntimeError(
                    f"Quantification différente pour la sortie {output_index}."
                )

            ya = original.get_tensor(oa["index"])
            yb = modified.get_tensor(ob["index"])

            if not np.array_equal(ya, yb):
                if ya.shape != yb.shape:
                    raise RuntimeError("La forme de sortie a changé.")

                delta = np.abs(
                    ya.astype(np.int32) - yb.astype(np.int32)
                )

                raise RuntimeError(
                    f"Test {test_index}, sortie {output_index} : "
                    f"sorties différentes, écart INT8 maximal={delta.max()}."
                )

        print(f"[OK] Test {test_index} : sorties strictement identiques.")


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    if DESTINATION.exists():
        raise FileExistsError(
            f"Le fichier existe déjà : {DESTINATION}\n"
            "Renomme-le ou supprime-le explicitement avant de recommencer."
        )

    original_bytes = SOURCE.read_bytes()
    root = schema.Model.GetRootAsModel(original_bytes, 0)
    model = schema.ModelT.InitFromObj(root)

    patched_ops = []
    opcode_indices = set()

    for sg_index, graph in enumerate(model.subgraphs):
        for op_index, op in enumerate(graph.operators):
            opcode = model.operatorCodes[op.opcodeIndex]
            code = max(
                opcode.builtinCode,
                opcode.deprecatedBuiltinCode,
            )

            if code != schema.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
                continue

            options = op.builtinOptions

            if options is None:
                raise RuntimeError("Options de redimensionnement absentes.")

            input_tensor = graph.tensors[int(op.inputs[0])]
            output_tensor = graph.tensors[int(op.outputs[0])]

            src_shape = tuple(int(v) for v in input_tensor.shape)
            dst_shape = tuple(int(v) for v in output_tensor.shape)

            if (
                len(src_shape) != 4
                or len(dst_shape) != 4
                or src_shape[0] != dst_shape[0]
                or src_shape[3] != dst_shape[3]
                or dst_shape[1] != 2 * src_shape[1]
                or dst_shape[2] != 2 * src_shape[2]
            ):
                raise RuntimeError(
                    f"Opération {op_index} : agrandissement NHWC ×2 "
                    f"non confirmé : {src_shape} -> {dst_shape}."
                )

            if (
                input_tensor.type != schema.TensorType.INT8
                or output_tensor.type != schema.TensorType.INT8
            ):
                raise RuntimeError("Les tenseurs doivent être INT8.")

            if opcode.version != 3:
                raise RuntimeError(
                    f"Version inattendue : {opcode.version}, attendu : 3."
                )

            if options.alignCorners or not options.halfPixelCenters:
                raise RuntimeError(
                    "Options inattendues : alignCorners doit être False "
                    "et halfPixelCenters doit être True."
                )

            options.halfPixelCenters = False
            opcode_indices.add(op.opcodeIndex)
            patched_ops.append((sg_index, op_index))

            print(
                f"[PATCH] Sous-graphe {sg_index}, opération {op_index} : "
                f"{src_shape} -> {dst_shape}"
            )

    if len(patched_ops) != 2:
        raise RuntimeError(
            f"Deux redimensionnements attendus, {len(patched_ops)} trouvés."
        )

    for index in opcode_indices:
        model.operatorCodes[index].version = 2

    model.signatureDefs = []

    builder = flatbuffers.Builder(0)
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    modified_bytes = bytes(builder.Output())

    original = create_interpreter(path=SOURCE)
    modified = create_interpreter(content=modified_bytes)

    compare_models(original, modified)

    DESTINATION.write_bytes(modified_bytes)

    print(f"\n[OK] Original conservé : {SOURCE}")
    print(f"[OK] Modèle modifié    : {DESTINATION}")
    print(f"[OK] Taille           : {len(modified_bytes)} octets")
    print(
        "\nCes tests sont un contrôle préliminaire. "
        "Valider ensuite sur des images réelles et sur le Coral."
    )


if __name__ == "__main__":
    main()
