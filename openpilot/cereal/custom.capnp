using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ReprojectState @0x81c2f05a394cf4af {
  # reprojectd: the 3X cameras reprojected into the comma 4 geometry, one per reprojected frame (20 Hz)
  frameId @0 :UInt32;        # the narrow camera frame the composite was built from
  stageMs @1 :Float32;       # GPU stage time for this frame
  rotation @2 :List(Float32); # applied narrow->wide rotation, rotvec in radians (x=pitch, y=yaw, z=roll)
  fitted @3 :Bool;           # the applied rotation is reprojectcalibd's fit (calibrationd holds until then)
}

struct ReprojectFit @0xaedffd8f31e7b55d {
  # reprojectcalibd: the narrow->wide rotation fit, 2 Hz and on every change
  enum Status {
    waiting @0;   # no fit accepted yet; why says what it waits for
    fitting @1;   # at least one fit accepted, more to come
    building @2;  # converged, the lookup tables are being built
    fitted @3;    # done; mean is the result reprojectd swaps in
  }
  enum Why {
    none @0;
    cameras @1;
    model @2;
    speed @3;
    straight @4;
    pair @5;
    features @6;  # the last frame had too little texture to fit
  }
  status @0 :Status;
  why @1 :Why;
  pct @2 :UInt8;             # progress of the fit, 0-100
  mean @3 :List(Float32);    # component-wise median of the accepted fits, rotvec in radians
  lastFrameId @4 :UInt32;    # the narrow camera frame the last fit ran on
  lastAccepted @5 :Bool;
}

struct ReprojectOutlines @0xf35cc4560bbf6ec2 {
  # reprojectd: the debug view's geometry (blend band, model inputs, comma 4 frames) as polygons in the px of each frame the
  # road view can show, republished when the applied rotation or the calibration moves. Derived from reprojectState and
  # extrinsicsCalibration, so not logged: a replay recomputes it.
  struct Item {
    frame @0 :Text;            # device_narrow | device_wide | c4_narrow | c4_wide | model_narrow | model_wide
    name @1 :Text;             # legend entry
    colour @2 :Text;           # amber | green | blue | purple
    points @3 :List(Float32);  # x0, y0, x1, y1, ...: a closed polygon, or a band's outer edge
    inner @4 :List(Float32);   # a band's inner edge, same point count; empty for a polygon
    visible @5 :Bool;          # any of it lands inside the frame
  }
  items @0 :List(Item);
}

struct CustomReserved3 @0xda96579883444c35 {
}

struct CustomReserved4 @0x80ae746ee2596b11 {
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
