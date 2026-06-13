# Runtime localisation direction

Runtime localisation is the intended second phase of FiduMap. It is not currently implemented as a separate complete module, but the map builder output is designed to support it directly.

## Intended workflow

1. Load the exported marker-corner CSV.
2. Detect markers in the current camera image.
3. For each detected marker ID, retrieve the corresponding world-frame marker corners.
4. Build 2D-3D correspondences:

   ```text
   detected image corner pixel <-> saved world-frame marker corner
   ```

5. Run PnP with RANSAC.
6. Refine the camera pose with reprojection residuals.
7. Output the camera pose in the marker-map/world frame.

## Why exported corners matter

The map builder internally optimises rigid marker poses, but runtime localisation benefits from direct world-frame corner coordinates. Exporting explicit corners makes it straightforward to feed detected image corners into a standard PnP/RANSAC pipeline.
