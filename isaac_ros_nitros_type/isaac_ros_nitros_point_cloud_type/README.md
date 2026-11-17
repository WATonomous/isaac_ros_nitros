# Point-cloud field preservation

Buffer-based NitrosPointCloud messages retain the complete PointCloud2 `fields`,
`is_dense`, dimensions, strides, endianness, and header. The ROS adapter validates
field bounds and payload size and copies all `row_step * height` bytes, including
organized row padding. It does not infer RGB from a four-float stride; XYZI is a
valid distinct layout. Consumers must validate the field types they support.

The builder's original `WithPoints(pointer, count, use_color)` remains packed XYZ
or XYZRGB. The overload `WithPoints(pointer, pointcloud_layout)` takes metadata from
a PointCloud2 and wraps an existing GPU allocation. Pair it with `WithHeader` for the
acquisition timestamp and frame. As with the original builder, ownership transfers
to the message unless a release callback is provided. Consumers use a read handle
on the stream that actually reads the device buffer.

Legacy GXF conversion is opt-in with `-DNITROS_POINT_CLOUD_GXF_COMPAT_MODE=ON`.
It is disabled by default: those message libraries export bundled protobuf symbols
that conflict with BEVFusion's protobuf library. Buffer-based Managed NITROS retains
GPU transport without linking them. A package-specific macro prevents other NITROS
types' legacy flags from accidentally enabling point-cloud GXF code.

Legacy GXF point-cloud messages support only packed XYZ/XYZRGB. Converting intensity,
custom fields, or padded layouts to that representation throws rather than silently
discarding or relabeling data. Buffer-based Managed NITROS consumers do not require
that conversion.
