(module
  ;; Five normalized voice features are combined locally. The weights mirror
  ;; the product explanation shown in the UI and intentionally avoid claiming
  ;; a medical or psychological diagnosis.
  (func (export "score")
    (param $pace f32)
    (param $pitch f32)
    (param $jitter f32)
    (param $pause f32)
    (param $volume f32)
    (result f32)
    local.get $pace
    f32.const 0.30
    f32.mul
    local.get $pitch
    f32.const 0.25
    f32.mul
    f32.add
    local.get $jitter
    f32.const 0.20
    f32.mul
    f32.add
    local.get $pause
    f32.const 0.15
    f32.mul
    f32.add
    local.get $volume
    f32.const 0.10
    f32.mul
    f32.add))
