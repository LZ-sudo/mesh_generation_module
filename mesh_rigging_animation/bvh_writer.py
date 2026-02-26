#!/usr/bin/env python3
"""
BVH skeleton hierarchy writer for CMU mocap format.

Provides write_bvh_hierarchy(), which emits the HIERARCHY section of a BVH
file.  The motion (MOTION) section is written by the calling pipeline
(c3d_to_bvh.write_bvh_from_joint_angles).

Skeleton summary:
  - Full CMU-style body: Hips, legs, spine, head, both arms
  - Only Spine1 (chest IMU) and the right arm chain are animated
  - All other joints are static (identity rotation)
  - Bone offsets are in centimeters (BVH standard)
  - Rotation order: ZYX (Zrotation Yrotation Xrotation)
"""


def write_bvh_hierarchy(f):
    """
    Write BVH HIERARCHY section matching CMU mocap format.

    Creates a CMU-style full-body skeleton compatible with existing animation assets:
    - Uses ZYX rotation order (Zrotation Yrotation Xrotation) to match reference BVHs
    - CMU bone naming: LowerBack, Spine, Spine1, RightArm, RightForeArm, RightHand
    - Spine chain: Hips -> LowerBack -> Spine -> Spine1 (Spine1 has IMU chest data)
    - Head: Neck -> Neck1 -> Head (dummy, static)
    - Right arm: RightShoulder -> RightArm -> RightForeArm -> RightHand (IMU data)
    - Left arm: LeftShoulder -> LeftArm (dummy, static)
    - Legs: LHipJoint/RHipJoint with full leg chains (dummy, static)

    This structure matches animation_assets/*.bvh format for consistency.
    Only the right arm and chest (Spine1) are animated from IMU data.

    Bone offsets are in centimeters (BVH standard units).
    """
    hierarchy = """HIERARCHY
ROOT Hips
{
\tOFFSET 0.00 0.00 0.00
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\tJOINT LHipJoint
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT LeftUpLeg
\t\t{
\t\t\tOFFSET 1.59 -1.84 0.72
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT LeftLeg
\t\t\t{
\t\t\t\tOFFSET 2.51 -6.88 0.00
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT LeftFoot
\t\t\t\t{
\t\t\t\t\tOFFSET 2.63 -7.23 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT LeftToeBase
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.24 -0.65 1.73
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.00 -0.00 0.93
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT RHipJoint
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT RightUpLeg
\t\t{
\t\t\tOFFSET -1.51 -1.84 0.72
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT RightLeg
\t\t\t{
\t\t\t\tOFFSET -2.55 -6.99 0.00
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT RightFoot
\t\t\t\t{
\t\t\t\t\tOFFSET -2.66 -7.31 0.00
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT RightToeBase
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -0.23 -0.63 2.04
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET -0.00 -0.00 1.07
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT LowerBack
\t{
\t\tOFFSET 0 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT Spine
\t\t{
\t\t\tOFFSET -0.03 1.86 -0.11
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT Spine1
\t\t\t{
\t\t\t\tOFFSET 0.01 1.86 0.04
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT Neck
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT Neck1
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -0.02 1.81 0.09
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT Head
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.06 1.76 -0.38
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET 0.02 1.83 -0.14
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT LeftShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT LeftArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 3.47 1.51 0.14
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT LeftForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 4.78 -0.00 0.00
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT LeftHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET 3.59 -0.00 -0.00
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tJOINT LeftFingerBase
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tJOINT LeftHandIndex1
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET 0.66 -0.00 0.00
\t\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\t\tOFFSET 0.53 -0.00 0.00
\t\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\tJOINT LThumb
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET 0.54 -0.00 0.54
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT RightShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT RightArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -3.32 1.61 0.35
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT RightForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET -4.49 -0.00 0.00
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT RightHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET -3.71 -0.00 0.00
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tJOINT RightFingerBase
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tJOINT RightHandIndex1
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET -0.45 -0.00 0.00
\t\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\t\tOFFSET -0.36 -0.00 0.00
\t\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\tJOINT RThumb
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0 0 0
\t\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\t\tOFFSET -0.37 -0.00 0.37
\t\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
}
"""
    f.write(hierarchy)
