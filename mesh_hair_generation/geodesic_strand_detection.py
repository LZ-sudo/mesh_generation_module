"""
Geodesic-Based Hair Strand Detection

Detects hair strand paths on continuous mesh hair assets using geodesic distance
computation. This approach works for MPFB-style hair meshes that are single
continuous triangle meshes without separate UV islands.

Algorithm:
1. Extract mesh data (vertices, faces) from Blender object
2. Identify scalp vertices (top N% by Z-height)
3. Compute geodesic distance from scalp using Dijkstra on mesh edges
4. Find hair tip candidates (top N% by geodesic distance)
5. Cluster tips spatially to reduce strand count
6. Trace strand paths from tips back to scalp
7. Generate bone definitions along each strand path

Dependencies:
    - numpy: Array operations
    - scipy (optional): Spatial clustering (KDTree) - has greedy fallback

Usage:
    from geodesic_strand_detection import (
        GeodesicConfig, detect_strands_geodesic, extract_mesh_for_geodesic
    )

    vertices, faces = extract_mesh_for_geodesic(hair_obj)
    strands = detect_strands_geodesic(vertices, faces, config)
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path

# Optional scipy for KDTree clustering (has greedy fallback)
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None


@dataclass
class GeodesicConfig:
    """Configuration for geodesic-based hair strand detection."""

    # Scalp detection
    scalp_percentile: float = 12.0
    """Top N% of vertices by Z-height are considered scalp region."""

    # Tip detection
    tip_percentile: float = 92.0
    """Top N% of vertices by geodesic distance are tip candidates."""

    tip_cluster_distance: float = 0.025
    """Meters - merge tip candidates closer than this distance."""

    # Strand filtering
    min_strand_length: float = 0.03
    """Meters - minimum strand length to generate bones for."""

    max_strands: int = 60
    """Maximum number of strand paths to generate."""

    # Bone generation
    bones_per_10cm: float = 3.0
    """Number of bones per 10cm of strand length."""

    min_bones_per_strand: int = 2
    """Minimum bones per strand."""

    max_bones_per_strand: int = 8
    """Maximum bones per strand."""

    # Path tracing
    path_smoothing_iterations: int = 2
    """Number of smoothing passes on traced paths."""

    # Vertex weighting
    weight_falloff: float = 2.0
    """Falloff exponent for vertex weight calculation."""


@dataclass
class HairStrand:
    """Represents a detected hair strand path."""

    path_coords: List[Tuple[float, float, float]]
    """Ordered coordinates from root (scalp) to tip."""

    path_vertex_indices: List[int]
    """Vertex indices along the path."""

    length: float
    """Total length of the strand in meters."""

    root_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Position of strand root (scalp attachment)."""

    tip_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Position of strand tip."""

    direction: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, -1.0))
    """Normalized direction from root to tip."""

    nearby_vertices: List[int] = field(default_factory=list)
    """Vertices near this strand for weight assignment."""

    @property
    def bone_count(self) -> int:
        """Calculate recommended number of bones based on length."""
        # 3 bones per 10cm
        count = max(2, int(self.length * 100 * 0.3))
        return min(count, 8)


def check_dependencies() -> Tuple[bool, str]:
    """
    Check if required dependencies are available.

    scipy is optional for tip clustering - we have a greedy fallback.

    Returns:
        Tuple of (available, message)
    """
    messages = []
    messages.append("geodesic: Dijkstra on mesh edges")

    if SCIPY_AVAILABLE:
        messages.append("scipy: available (KDTree clustering)")
    else:
        messages.append("scipy: not available (using greedy clustering)")

    # Always return True since we have fallbacks for everything
    return True, "; ".join(messages)


def extract_mesh_for_geodesic(hair_obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract vertex and face data from a Blender mesh object.

    Args:
        hair_obj: Blender mesh object

    Returns:
        Tuple of:
        - vertices: (N, 3) array of world-space vertex positions
        - faces: (M, 3) array of triangle face indices
        - vertex_indices: (N,) array mapping local index to original mesh index
    """
    mesh = hair_obj.data
    world_matrix = hair_obj.matrix_world

    # Extract vertices in world space
    num_verts = len(mesh.vertices)
    vertices = np.zeros((num_verts, 3), dtype=np.float64)

    for i, v in enumerate(mesh.vertices):
        world_pos = world_matrix @ v.co
        vertices[i] = [world_pos.x, world_pos.y, world_pos.z]

    # Extract faces (triangles only)
    # Count triangles first
    num_tris = sum(1 for p in mesh.polygons if len(p.vertices) == 3)
    num_quads = sum(1 for p in mesh.polygons if len(p.vertices) == 4)
    total_tris = num_tris + num_quads * 2

    faces = np.zeros((total_tris, 3), dtype=np.int32)
    face_idx = 0

    for poly in mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) == 3:
            faces[face_idx] = verts
            face_idx += 1
        elif len(verts) == 4:
            # Triangulate quad
            faces[face_idx] = [verts[0], verts[1], verts[2]]
            faces[face_idx + 1] = [verts[0], verts[2], verts[3]]
            face_idx += 2
        # Skip n-gons with more than 4 vertices

    # Trim faces array if we skipped any n-gons
    faces = faces[:face_idx]

    # Vertex indices mapping (identity for now, but useful for subset operations)
    vertex_indices = np.arange(num_verts, dtype=np.int32)

    return vertices, faces, vertex_indices


def _identify_scalp_vertices(
    vertices: np.ndarray,
    config: GeodesicConfig
) -> np.ndarray:
    """
    Identify scalp vertices based on Z-height.

    For simple hair that grows downward, scalp vertices are at the top
    of the mesh (highest Z values).

    Args:
        vertices: (N, 3) vertex positions
        config: Detection configuration

    Returns:
        Array of vertex indices in the scalp region
    """
    z_coords = vertices[:, 2]

    # Find Z threshold for top N%
    threshold = np.percentile(z_coords, 100 - config.scalp_percentile)

    # Get vertices above threshold
    scalp_mask = z_coords >= threshold
    scalp_indices = np.where(scalp_mask)[0]

    return scalp_indices


def _compute_geodesic_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Compute geodesic distance from source vertices using Dijkstra's algorithm.

    Computes shortest path distances along mesh edges, which approximates
    geodesic distance for sufficiently dense meshes.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        source_indices: Indices of source vertices (scalp)
        verbose: Enable verbose output

    Returns:
        (N,) array of geodesic distances, or None if computation fails
    """
    return _compute_geodesic_distance_dijkstra(vertices, faces, source_indices, verbose)


def _build_weighted_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build weighted adjacency list from mesh faces.

    Each edge is weighted by the Euclidean distance between vertices.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices

    Returns:
        Dict mapping vertex index to list of (neighbor_index, distance) tuples
    """
    num_verts = len(vertices)
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_verts)}

    # Track edges we've already added to avoid duplicates
    seen_edges: Set[Tuple[int, int]] = set()

    for face in faces:
        # Add edges for each pair of vertices in the triangle
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]

        for v0, v1 in edges:
            # Normalize edge direction for deduplication
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Calculate edge length (Euclidean distance)
            dist = np.linalg.norm(vertices[v0] - vertices[v1])

            # Add bidirectional edges
            adjacency[v0].append((v1, dist))
            adjacency[v1].append((v0, dist))

    return adjacency


def _compute_geodesic_distance_dijkstra(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_indices: np.ndarray,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Compute approximate geodesic distance using Dijkstra's algorithm on mesh edges.

    This is a fallback when potpourri3d is not available. It computes shortest
    path distances along mesh edges, which approximates geodesic distance for
    sufficiently dense meshes.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        source_indices: Indices of source vertices (scalp)
        verbose: Enable verbose output

    Returns:
        (N,) array of distances from nearest source, or None if computation fails
    """
    num_verts = len(vertices)

    if verbose:
        print("    Using Dijkstra fallback (no potpourri3d)")

    try:
        # Build weighted adjacency graph
        adjacency = _build_weighted_adjacency(vertices, faces)

        # Initialize distances to infinity
        distances = np.full(num_verts, np.inf, dtype=np.float64)

        # Priority queue: (distance, vertex_index)
        # Initialize with all source vertices at distance 0
        heap: List[Tuple[float, int]] = []
        for src_idx in source_indices:
            distances[src_idx] = 0.0
            heapq.heappush(heap, (0.0, src_idx))

        # Dijkstra's algorithm
        visited = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            # Skip if we found a shorter path already
            if dist > distances[current]:
                continue

            # Relax edges to neighbors
            for neighbor, edge_dist in adjacency[current]:
                new_dist = dist + edge_dist

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        # Check for unreachable vertices (disconnected mesh components)
        unreachable = np.sum(np.isinf(distances))
        if unreachable > 0 and verbose:
            print(f"    Warning: {unreachable} vertices unreachable from scalp")

        # Replace infinity with max finite distance for unreachable vertices
        max_finite = np.max(distances[np.isfinite(distances)])
        distances[np.isinf(distances)] = max_finite

        return distances

    except Exception as e:
        print(f"  Dijkstra distance computation failed: {e}")
        return None


def _find_tip_candidates(
    vertices: np.ndarray,
    geodesic_dist: np.ndarray,
    config: GeodesicConfig
) -> np.ndarray:
    """
    Find hair tip candidate vertices based on geodesic distance.

    Tips are vertices with high geodesic distance from the scalp.

    Args:
        vertices: (N, 3) vertex positions
        geodesic_dist: (N,) geodesic distances from scalp
        config: Detection configuration

    Returns:
        Array of tip candidate vertex indices
    """
    # Find distance threshold for top N%
    threshold = np.percentile(geodesic_dist, config.tip_percentile)

    # Get vertices above threshold
    tip_mask = geodesic_dist >= threshold
    tip_indices = np.where(tip_mask)[0]

    return tip_indices


def _cluster_tips(
    vertices: np.ndarray,
    tip_indices: np.ndarray,
    geodesic_dist: np.ndarray,
    config: GeodesicConfig
) -> List[int]:
    """
    Cluster nearby tip candidates and select representative tips.

    Uses spatial clustering to merge tips that are close together,
    keeping the one with maximum geodesic distance in each cluster.

    Args:
        vertices: (N, 3) vertex positions
        tip_indices: Indices of tip candidate vertices
        geodesic_dist: (N,) geodesic distances
        config: Detection configuration

    Returns:
        List of representative tip vertex indices
    """
    if len(tip_indices) == 0:
        return []

    if len(tip_indices) == 1:
        return [tip_indices[0]]

    tip_positions = vertices[tip_indices]
    tip_distances = geodesic_dist[tip_indices]

    # Use simple greedy clustering if scipy not available
    if not SCIPY_AVAILABLE:
        return _cluster_tips_greedy(tip_indices, tip_positions, tip_distances, config)

    # Build KD-tree for spatial queries
    tree = KDTree(tip_positions)

    # Find clusters using distance threshold
    cluster_distance = config.tip_cluster_distance

    # Track which tips have been assigned to clusters
    assigned = set()
    representatives = []

    # Sort by geodesic distance (descending) to prioritize furthest tips
    sorted_indices = np.argsort(-tip_distances)

    for idx in sorted_indices:
        tip_idx = tip_indices[idx]

        if tip_idx in assigned:
            continue

        # Find all tips within cluster distance
        pos = tip_positions[idx]
        neighbors = tree.query_ball_point(pos, cluster_distance)

        # Mark all neighbors as assigned
        for n in neighbors:
            assigned.add(tip_indices[n])

        # This tip is the representative (has highest geodesic distance in cluster)
        representatives.append(tip_idx)

        # Stop if we have enough strands
        if len(representatives) >= config.max_strands:
            break

    return representatives


def _cluster_tips_greedy(
    tip_indices: np.ndarray,
    tip_positions: np.ndarray,
    tip_distances: np.ndarray,
    config: GeodesicConfig
) -> List[int]:
    """
    Simple greedy clustering fallback when scipy is not available.
    """
    cluster_dist_sq = config.tip_cluster_distance ** 2

    # Sort by geodesic distance (descending)
    sorted_order = np.argsort(-tip_distances)

    representatives = []
    used = set()

    for idx in sorted_order:
        if idx in used:
            continue

        tip_idx = tip_indices[idx]
        pos = tip_positions[idx]

        # Check if too close to existing representative
        too_close = False
        for rep_idx in representatives:
            rep_local_idx = np.where(tip_indices == rep_idx)[0][0]
            rep_pos = tip_positions[rep_local_idx]
            dist_sq = np.sum((pos - rep_pos) ** 2)
            if dist_sq < cluster_dist_sq:
                too_close = True
                break

        if not too_close:
            representatives.append(tip_idx)
            used.add(idx)

            # Mark nearby tips as used
            for other_idx in range(len(tip_indices)):
                if other_idx in used:
                    continue
                other_pos = tip_positions[other_idx]
                dist_sq = np.sum((pos - other_pos) ** 2)
                if dist_sq < cluster_dist_sq:
                    used.add(other_idx)

        if len(representatives) >= config.max_strands:
            break

    return representatives


def _build_adjacency(
    faces: np.ndarray,
    num_vertices: int
) -> Dict[int, Set[int]]:
    """
    Build vertex adjacency map from faces.

    Args:
        faces: (M, 3) triangle indices
        num_vertices: Total number of vertices

    Returns:
        Dict mapping vertex index to set of adjacent vertex indices
    """
    adjacency = {i: set() for i in range(num_vertices)}

    for face in faces:
        v0, v1, v2 = face
        adjacency[v0].add(v1)
        adjacency[v0].add(v2)
        adjacency[v1].add(v0)
        adjacency[v1].add(v2)
        adjacency[v2].add(v0)
        adjacency[v2].add(v1)

    return adjacency


def _trace_path_to_scalp(
    tip_idx: int,
    vertices: np.ndarray,
    geodesic_dist: np.ndarray,
    adjacency: Dict[int, Set[int]],
    scalp_indices: Set[int]
) -> List[int]:
    """
    Trace a path from tip vertex back to scalp following geodesic gradient.

    Uses greedy descent: at each vertex, move to the neighbor with
    smallest geodesic distance until reaching the scalp.

    Args:
        tip_idx: Starting tip vertex index
        vertices: (N, 3) vertex positions
        geodesic_dist: (N,) geodesic distances
        adjacency: Vertex adjacency map
        scalp_indices: Set of scalp vertex indices

    Returns:
        List of vertex indices from tip to root (reversed for root-to-tip order)
    """
    path = [tip_idx]
    current = tip_idx
    visited = {tip_idx}

    max_iterations = len(vertices)  # Safety limit

    for _ in range(max_iterations):
        # Check if we reached scalp
        if current in scalp_indices:
            break

        # Find neighbor with minimum geodesic distance
        neighbors = adjacency.get(current, set())
        if not neighbors:
            break

        best_neighbor = None
        best_dist = geodesic_dist[current]

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            neighbor_dist = geodesic_dist[neighbor]
            if neighbor_dist < best_dist:
                best_dist = neighbor_dist
                best_neighbor = neighbor

        if best_neighbor is None:
            # No unvisited neighbor with lower distance - we're stuck
            # Try to find any unvisited neighbor closer to scalp
            for neighbor in neighbors:
                if neighbor not in visited:
                    if best_neighbor is None or geodesic_dist[neighbor] < geodesic_dist[best_neighbor]:
                        best_neighbor = neighbor

            if best_neighbor is None:
                break

        path.append(best_neighbor)
        visited.add(best_neighbor)
        current = best_neighbor

    # Reverse path to go from root to tip
    return list(reversed(path))


def _smooth_path(
    path_indices: List[int],
    vertices: np.ndarray,
    iterations: int = 2
) -> List[Tuple[float, float, float]]:
    """
    Smooth path coordinates using simple averaging.

    Args:
        path_indices: Vertex indices along path
        vertices: (N, 3) vertex positions
        iterations: Number of smoothing passes

    Returns:
        Smoothed path as list of (x, y, z) coordinates
    """
    if len(path_indices) < 3:
        return [tuple(vertices[i]) for i in path_indices]

    # Get initial coordinates
    coords = np.array([vertices[i] for i in path_indices])

    for _ in range(iterations):
        smoothed = coords.copy()
        # Don't smooth endpoints
        for i in range(1, len(coords) - 1):
            smoothed[i] = (coords[i-1] + coords[i] + coords[i+1]) / 3.0
        coords = smoothed

    return [tuple(c) for c in coords]


def _calculate_path_length(coords: List[Tuple[float, float, float]]) -> float:
    """Calculate total length of a path."""
    if len(coords) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i-1][0]
        dy = coords[i][1] - coords[i-1][1]
        dz = coords[i][2] - coords[i-1][2]
        length += np.sqrt(dx*dx + dy*dy + dz*dz)

    return length


def _assign_nearby_vertices(
    strand: HairStrand,
    vertices: np.ndarray,
    all_strand_paths: List[List[int]],
    strand_idx: int
) -> List[int]:
    """
    Assign vertices to the nearest strand for weight calculation.

    Uses a simple approach: for each vertex not on any path,
    find which strand path it's closest to.

    This is called per-strand to find vertices that should be
    weighted to this strand.
    """
    # For now, just return vertices along the path
    # Full implementation would do spatial assignment
    return strand.path_vertex_indices.copy()


def detect_strands_geodesic(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Optional[GeodesicConfig] = None,
    verbose: bool = False
) -> Optional[List[HairStrand]]:
    """
    Detect hair strand paths using geodesic distance computation.

    This is the main entry point for geodesic-based strand detection.

    Args:
        vertices: (N, 3) array of vertex positions in world space
        faces: (M, 3) array of triangle face indices
        config: Detection configuration (uses defaults if None)
        verbose: Enable verbose output

    Returns:
        List of HairStrand objects, or None if detection fails
    """
    # Check dependencies
    available, msg = check_dependencies()
    if not available:
        if verbose:
            print(f"  {msg}")
        return None

    if config is None:
        config = GeodesicConfig()

    num_verts = len(vertices)
    num_faces = len(faces)

    if verbose:
        print(f"  Geodesic strand detection: {num_verts} vertices, {num_faces} faces")

    # Validate mesh
    if num_verts < 10:
        if verbose:
            print("  Mesh too small for geodesic detection")
        return None

    if num_faces < 5:
        if verbose:
            print("  Too few faces for geodesic detection")
        return None

    # Step 1: Identify scalp vertices
    scalp_indices = _identify_scalp_vertices(vertices, config)
    if verbose:
        print(f"  Identified {len(scalp_indices)} scalp vertices (top {config.scalp_percentile}%)")

    if len(scalp_indices) < 3:
        if verbose:
            print("  Too few scalp vertices identified")
        return None

    # Step 2: Compute geodesic distance from scalp
    if verbose:
        print("  Computing geodesic distances...")

    geodesic_dist = _compute_geodesic_distance(vertices, faces, scalp_indices, verbose)
    if geodesic_dist is None:
        if verbose:
            print("  Geodesic computation failed")
        return None

    max_dist = np.max(geodesic_dist)
    if verbose:
        print(f"  Max geodesic distance: {max_dist*100:.1f}cm")

    # Step 3: Find tip candidates
    tip_candidates = _find_tip_candidates(vertices, geodesic_dist, config)
    if verbose:
        print(f"  Found {len(tip_candidates)} tip candidates")

    if len(tip_candidates) < 1:
        if verbose:
            print("  No tip candidates found")
        return None

    # Step 4: Cluster tips
    representative_tips = _cluster_tips(vertices, tip_candidates, geodesic_dist, config)
    if verbose:
        print(f"  Clustered to {len(representative_tips)} representative tips")

    if len(representative_tips) < 1:
        if verbose:
            print("  No representative tips after clustering")
        return None

    # Step 5: Build adjacency for path tracing
    adjacency = _build_adjacency(faces, num_verts)
    scalp_set = set(scalp_indices)

    # Step 6: Trace paths from tips to scalp
    if verbose:
        print("  Tracing strand paths...")

    strands = []
    all_paths = []

    for tip_idx in representative_tips:
        # Trace path
        path_indices = _trace_path_to_scalp(
            tip_idx, vertices, geodesic_dist, adjacency, scalp_set
        )

        if len(path_indices) < 2:
            continue

        # Smooth path coordinates
        path_coords = _smooth_path(
            path_indices, vertices, config.path_smoothing_iterations
        )

        # Calculate length
        length = _calculate_path_length(path_coords)

        if length < config.min_strand_length:
            continue

        # Calculate direction
        root = path_coords[0]
        tip = path_coords[-1]
        dx = tip[0] - root[0]
        dy = tip[1] - root[1]
        dz = tip[2] - root[2]
        dir_len = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dir_len > 0.0001:
            direction = (dx/dir_len, dy/dir_len, dz/dir_len)
        else:
            direction = (0.0, 0.0, -1.0)

        strand = HairStrand(
            path_coords=path_coords,
            path_vertex_indices=path_indices,
            length=length,
            root_position=root,
            tip_position=tip,
            direction=direction,
            nearby_vertices=path_indices.copy()
        )

        strands.append(strand)
        all_paths.append(path_indices)

    if verbose:
        print(f"  Generated {len(strands)} valid strands")
        if strands:
            lengths = [s.length * 100 for s in strands]
            print(f"  Strand lengths: min={min(lengths):.1f}cm, "
                  f"max={max(lengths):.1f}cm, avg={sum(lengths)/len(lengths):.1f}cm")

    return strands if strands else None


def assign_vertices_to_strands(
    vertices: np.ndarray,
    strands: List[HairStrand],
    config: Optional[GeodesicConfig] = None
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Assign all mesh vertices to nearby strands with weights.

    Each vertex is assigned to the nearest strand(s) based on
    distance to the strand path.

    Args:
        vertices: (N, 3) vertex positions
        strands: List of detected strands
        config: Configuration (for weight falloff)

    Returns:
        Dict mapping strand index to list of (vertex_index, weight) tuples
    """
    if config is None:
        config = GeodesicConfig()

    if not strands:
        return {}

    num_verts = len(vertices)

    # For each vertex, find distance to each strand
    vertex_strand_distances = np.full((num_verts, len(strands)), np.inf)

    for strand_idx, strand in enumerate(strands):
        path_coords = np.array(strand.path_coords)

        for v_idx in range(num_verts):
            v_pos = vertices[v_idx]

            # Find minimum distance to any segment of the strand path
            min_dist = np.inf
            for i in range(len(path_coords) - 1):
                p0 = path_coords[i]
                p1 = path_coords[i + 1]

                # Distance from point to line segment
                segment = p1 - p0
                seg_len_sq = np.dot(segment, segment)

                if seg_len_sq < 0.00001:
                    dist = np.linalg.norm(v_pos - p0)
                else:
                    t = max(0.0, min(1.0, np.dot(v_pos - p0, segment) / seg_len_sq))
                    projection = p0 + t * segment
                    dist = np.linalg.norm(v_pos - projection)

                min_dist = min(min_dist, dist)

            vertex_strand_distances[v_idx, strand_idx] = min_dist

    # Assign each vertex to nearest strand with weight based on distance
    strand_vertices = {i: [] for i in range(len(strands))}

    for v_idx in range(num_verts):
        distances = vertex_strand_distances[v_idx]
        nearest_strand = np.argmin(distances)
        min_dist = distances[nearest_strand]

        # Weight based on distance (closer = higher weight)
        # Using inverse distance with falloff
        if min_dist < 0.001:
            weight = 1.0
        else:
            weight = 1.0 / (1.0 + (min_dist * 10) ** config.weight_falloff)

        if weight > 0.01:  # Minimum weight threshold
            strand_vertices[nearest_strand].append((v_idx, weight))

    return strand_vertices


def strands_to_bone_data(
    strands: List[HairStrand],
    bone_prefix: str = "hair",
    config: Optional[GeodesicConfig] = None
) -> List[Dict]:
    """
    Convert strand paths to bone definition data.

    Args:
        strands: List of detected strands
        bone_prefix: Prefix for bone names
        config: Configuration for bone count

    Returns:
        List of bone definition dicts compatible with MPFB skeleton format
    """
    if config is None:
        config = GeodesicConfig()

    bones = []

    for strand_idx, strand in enumerate(strands):
        path = strand.path_coords

        # Determine number of bones for this strand
        bone_count = int(strand.length * 100 * config.bones_per_10cm / 10.0)
        bone_count = max(config.min_bones_per_strand,
                        min(config.max_bones_per_strand, bone_count))

        # Resample path to bone count + 1 points
        resampled = _resample_path(path, bone_count + 1)

        # Create bones along resampled path
        parent_name = "head"  # First bone parents to head

        for bone_idx in range(bone_count):
            bone_name = f"{bone_prefix}_{strand_idx}_{bone_idx}"

            head_pos = resampled[bone_idx]
            tail_pos = resampled[bone_idx + 1]

            bone_data = {
                "name": bone_name,
                "head": list(head_pos),
                "tail": list(tail_pos),
                "parent": parent_name,
                "strand_index": strand_idx,
                "bone_index": bone_idx
            }

            bones.append(bone_data)
            parent_name = bone_name  # Next bone parents to this one

    return bones


def _resample_path(
    path: List[Tuple[float, float, float]],
    num_points: int
) -> List[Tuple[float, float, float]]:
    """
    Resample a path to have a specific number of evenly-spaced points.

    Args:
        path: Original path coordinates
        num_points: Desired number of points

    Returns:
        Resampled path coordinates
    """
    if len(path) < 2:
        return path

    if num_points <= 2:
        return [path[0], path[-1]]

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dz = path[i][2] - path[i-1][2]
        distances.append(distances[-1] + np.sqrt(dx*dx + dy*dy + dz*dz))

    total_length = distances[-1]
    if total_length < 0.0001:
        return [path[0]] * num_points

    # Generate evenly spaced distances
    target_distances = np.linspace(0, total_length, num_points)

    # Interpolate positions
    resampled = []
    path_idx = 0

    for target_dist in target_distances:
        # Find segment containing target distance
        while path_idx < len(distances) - 1 and distances[path_idx + 1] < target_dist:
            path_idx += 1

        if path_idx >= len(path) - 1:
            resampled.append(path[-1])
            continue

        # Interpolate within segment
        seg_start = distances[path_idx]
        seg_end = distances[path_idx + 1]
        seg_length = seg_end - seg_start

        if seg_length < 0.0001:
            t = 0.0
        else:
            t = (target_dist - seg_start) / seg_length

        p0 = path[path_idx]
        p1 = path[path_idx + 1]

        interpolated = (
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2])
        )
        resampled.append(interpolated)

    return resampled
