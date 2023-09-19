// Pixels
@group(0) @binding(0)  
  var<storage, read_write> pixels : array<vec4f>;

// Uniforms
@group(1) @binding(0) 
  var<uniform> rez : f32;

@group(1) @binding(1) 
  var<uniform> time : f32;

@group(1) @binding(2) 
  var<uniform> count : u32;

@group(1) @binding(3) 
  var<uniform> alpha : f32;

@group(1) @binding(4) 
  var<uniform> beta : f32;

@group(1) @binding(5)  
  var<uniform> radius : f32;

@group(1) @binding(6)  
  var<uniform> velocity : f32;

// Other buffers
@group(2) @binding(0)  
  var<storage, read_write> positions : array<vec2f>;

@group(2) @binding(1)  
  var<storage, read_write> headings : array<vec2f>;

fn r(n: f32) -> f32 {
  let x = sin(n) * 43758.5453;
  return fract(x);
}

fn index(p: vec2f) -> i32 {
  return i32(p.x) + i32(p.y) * i32(rez);
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id : vec3u) {
  let seed = f32(id.x)/f32(count);
  var p = vec2(r(seed), r(seed + 0.1));
  p *= rez;
  positions[id.x] = p;
  headings[id.x] = normalize(vec2(r(f32(id.x+1)), r(f32(id.x + 2))) - 0.5);
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) {
  var p = positions[id.x];
  var h = headings[id.x];

  let a = normalize(h);
  var r = 0;
  var l = 0;


  for(var i = 0u; i < count; i++) {
    if(i == id.x)
    {
      continue;
    }
    
    let other = positions[i];
    
    if(distance(p, other) > radius) 
    {
      continue;
    }
    
    let b = normalize(other - p);
    let is_r = is_on_right_side(a, b);
    r += select(0, 1, is_r); // false, true, condition
    l += select(1, 0, is_r);
  }

  let n = l + r;
  let density = f32(count) / pow(rez, 2);
  let relative_n = f32(n) / (density * 7000.);

  let sign_r_l = sign(r - l);
  let delta_phi = alpha + beta * f32(n) * f32(sign_r_l);

  h *= rotate2d(delta_phi);

  p += h * velocity;

  // wrap
  p = (p + rez) % rez;

  positions[id.x] = p;
  headings[id.x] = h;

  let cos_h_c = (h.y + 1) / 2.;

  var c = hsl_to_rgb(vec3(((cos_h_c + f32(sign_r_l) * .8 + relative_n)) % 1., 1., .5));

  pixels[index(p)] = mix(pixels[index(p)], vec4(c, 1.), 0.3);
  // pixels[index(p)] = vec4(c, 1.);
}

@compute @workgroup_size(256)
fn fade(@builtin(global_invocation_id) id : vec3u) 
{
  pixels[id.x] *= 0.92;
}

fn is_on_right_side(a: vec2f, b: vec2f) -> bool 
{
  var a_rot_90 = vec2(a.y, -a.x);
  var dot_product = a_rot_90.x * b.x + a_rot_90.y * b.y;
  return dot_product < 0;
}

fn rotate2d(angle : f32) -> mat2x2<f32> 
{
    return mat2x2<f32>(cos(angle), -sin(angle),
                       sin(angle),  cos(angle));
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f
{
   let r = abs(hsl.x * 6.0 - 3.0) - 1.0;
   let g = 2.0 - abs(hsl.x * 6.0 - 2.0);
   let b = 2.0 - abs(hsl.x * 6.0 - 4.0);
   let c = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
   var rgb = vec3(r, g, b);
   return vec3((rgb - 0.5) * c + hsl.z);
}