"""
Standalone CoACD runner — executed via myenv Python by collision_mesh_generation.py.

Reads per-region mesh data (vertices + triangulated faces) from a JSON file,
runs CoACD convex decomposition on each region, and writes the resulting
convex parts back to a JSON file.

This script is intentionally separate from the Blender context so that the
coacd package (installed in myenv) can be used without needing to be available
inside Blender's bundled Python interpreter.
"""

import argparse
import json
import sys

import numpy as np


def _decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_vertices: int,
):
    """
    Reduce vertex count via voxel-grid decimation.

    Bins vertices into a uniform 3-D grid and retains one representative per
    occupied cell.  Faces whose every vertex survives are kept and remapped.
    Only numpy is required — no extra dependencies.

    Args:
        vertices: (N, 3) float array.
        faces: (M, 3) int array.
        max_vertices: Target upper bound on retained vertices.

    Returns:
        (new_vertices, new_faces) as numpy arrays.
    """
    if len(vertices) <= max_vertices:
        return vertices, faces

    bbox_min = vertices.min(axis=0)
    bbox_range = vertices.max(axis=0) - bbox_min + 1e-8

    grid_size = max(1, int(round(max_vertices ** (1 / 3))))

    normalised = (vertices - bbox_min) / bbox_range
    cell = np.floor(normalised * grid_size).astype(np.int32).clip(0, grid_size - 1)
    voxel_key = (
        cell[:, 0] * grid_size * grid_size
        + cell[:, 1] * grid_size
        + cell[:, 2]
    )

    # Keep the first vertex encountered in each voxel
    _, keep_indices = np.unique(voxel_key, return_index=True)
    keep_set = set(keep_indices.tolist())

    old_to_new = {old: new for new, old in enumerate(sorted(keep_indices))}
    new_verts = vertices[sorted(keep_indices)]

    new_faces = []
    for f in faces.tolist():
        if all(v in keep_set for v in f):
            new_faces.append([old_to_new[v] for v in f])

    return new_verts, np.array(new_faces, dtype=np.int32) if new_faces else np.empty((0, 3), dtype=np.int32)


def run_decomposition(
    input_path: str,
    output_path: str,
    threshold: float,
    max_vertices: int = 2000,
) -> None:
    """
    Decompose all regions in the input JSON with CoACD and write results.

    Each region is decimated to at most max_vertices before decomposition to
    keep processing time manageable for high-poly MPFB2 meshes.

    Args:
        input_path: Path to JSON with structure:
            {region_name: {"vertices": [[x,y,z], ...], "faces": [[i,j,k], ...]}}
        output_path: Path to write JSON with structure:
            {region_name: [{"vertices": [...], "faces": [...]}, ...]}
        threshold: CoACD concavity threshold (lower = tighter, more parts).
        max_vertices: Vertex cap applied before CoACD (default: 5000).
    """
    try:
        import coacd
    except ImportError:
        print(
            "ERROR: coacd is not installed in this environment.\n"
            "Install it with: pip install coacd",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        region_data = json.load(f)

    output: dict = {}

    for region_name, mesh_data in region_data.items():
        vertices = np.array(mesh_data["vertices"], dtype=np.float64)
        faces = np.array(mesh_data["faces"], dtype=np.int32)

        orig_vert_count = len(vertices)
        vertices, faces = _decimate_mesh(vertices, faces, max_vertices)

        if len(faces) < 4:
            print(f"  {region_name}: skipped after decimation (too few faces)")
            continue

        print(
            f"  {region_name}: {orig_vert_count} -> {len(vertices)} verts "
            f"({len(faces)} faces) after decimation"
        )

        mesh = coacd.Mesh(vertices, faces)
        # threshold is a normalized concavity ratio relative to each region's
        # own bounding box (real_metric=False is the default).  Using the
        # absolute-metre metric (real_metric=True) caused collision shapes
        # many times larger than the actual limbs for threshold=0.2m.
        parts = coacd.run_coacd(
            mesh,
            threshold=threshold,
            mcts_iterations=100,
            mcts_max_depth=3,
        )

        # parts is a list of (vertices_ndarray, faces_ndarray) tuples
        output[region_name] = [
            {
                "vertices": part[0].tolist(),
                "faces": part[1].tolist(),
            }
            for part in parts
        ]

        print(f"  {region_name}: {len(parts)} convex part(s)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"CoACD complete: {len(output)} regions processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CoACD approximate convex decomposition on body regions"
    )
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold (default: 0.05). Normalized ratio of each region's bounding box. Range: 0.01 (fine) to 0.3 (coarse).",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=2000,
        help="Vertex cap per region before CoACD (default: 2000).",
    )
    args = parser.parse_args()

    run_decomposition(args.input, args.output, args.threshold, args.max_vertices)
