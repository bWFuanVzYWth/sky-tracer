// Ported from drt-bench shaders/ap/techniques/drt/Oklab.glsl.
// Enhanced Reinhard preserves 18% middle gray; gamut mapping happens on
// constant Oklab hue rays and is monotone within each lightness slice.

const REINHARD_GAMUT_RGB_HEADROOM: f32 = 0.99999;
const REINHARD_GAMUT_MIDDLE_GRAY: f32 = 0.18;
const REINHARD_GAMUT_RED_ROW: vec3<f32> = vec3<f32>(4.0767416621, -3.3077115913, 0.2309699292);
const REINHARD_GAMUT_GREEN_ROW: vec3<f32> = vec3<f32>(-1.2684380046, 2.6097574011, -0.3413193965);
const REINHARD_GAMUT_BLUE_ROW: vec3<f32> = vec3<f32>(-0.0041960863, -0.7034186147, 1.7076147010);

fn reinhard_gamut_rgb_to_oklab(color: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    let m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    let s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    let lms = pow(vec3<f32>(l, m, s), vec3<f32>(1.0 / 3.0));
    return vec3<f32>(
        0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
        1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
        0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z,
    );
}

fn reinhard_gamut_oklab_to_rgb(color: vec3<f32>) -> vec3<f32> {
    let l_root = color.x + 0.3963377774 * color.y + 0.2158037573 * color.z;
    let m_root = color.x - 0.1055613458 * color.y - 0.0638541728 * color.z;
    let s_root = color.x - 0.0894841775 * color.y - 1.2914855480 * color.z;
    let lms = vec3<f32>(
        l_root * l_root * l_root,
        m_root * m_root * m_root,
        s_root * s_root * s_root,
    );
    return vec3<f32>(
        dot(REINHARD_GAMUT_RED_ROW, lms),
        dot(REINHARD_GAMUT_GREEN_ROW, lms),
        dot(REINHARD_GAMUT_BLUE_ROW, lms),
    );
}

fn reinhard_gamut_shoulder_coefficient(overexposure: f32) -> f32 {
    let scale = overexposure / (overexposure - REINHARD_GAMUT_MIDDLE_GRAY);
    return (scale * scale - 1.0) / REINHARD_GAMUT_MIDDLE_GRAY;
}

fn reinhard_gamut_map_lightness(lightness: f32, overexposure: f32) -> f32 {
    let brightness = lightness * lightness * lightness;
    let x = reinhard_gamut_shoulder_coefficient(overexposure) * brightness;
    let inverse_root = inverseSqrt(1.0 + x);
    let mapped_brightness = overexposure * x * inverse_root * inverse_root / (1.0 + inverse_root);
    return pow(mapped_brightness, 1.0 / 3.0);
}

fn reinhard_gamut_root_direction(hue: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.3963377774 * hue.x + 0.2158037573 * hue.y,
        -0.1055613458 * hue.x - 0.0638541728 * hue.y,
        -0.0894841775 * hue.x - 1.2914855480 * hue.y,
    );
}

fn reinhard_gamut_max_saturation(hue: vec2<f32>, direction: vec3<f32>) -> f32 {
    var k0: f32;
    var k1: f32;
    var k2: f32;
    var k3: f32;
    var k4: f32;
    var rgb_row: vec3<f32>;

    if (-1.88170328 * hue.x - 0.80936493 * hue.y > 1.0) {
        k0 = 1.19086277;
        k1 = 1.76576728;
        k2 = 0.59662641;
        k3 = 0.75515197;
        k4 = 0.56771245;
        rgb_row = REINHARD_GAMUT_RED_ROW;
    } else if (1.81444104 * hue.x - 1.19445276 * hue.y > 1.0) {
        k0 = 0.73956515;
        k1 = -0.45954404;
        k2 = 0.08285427;
        k3 = 0.12541070;
        k4 = 0.14503204;
        rgb_row = REINHARD_GAMUT_GREEN_ROW;
    } else {
        k0 = 1.35733652;
        k1 = -0.00915799;
        k2 = -1.15130210;
        k3 = -0.50559606;
        k4 = 0.00692167;
        rgb_row = REINHARD_GAMUT_BLUE_ROW;
    }

    let saturation = k0
        + k1 * hue.x
        + k2 * hue.y
        + k3 * hue.x * hue.x
        + k4 * hue.x * hue.y;
    let roots = vec3<f32>(1.0) + saturation * direction;
    let lms = roots * roots * roots;
    let first_lms = 3.0 * direction * roots * roots;
    let second_lms = 6.0 * direction * direction * roots;
    let f = dot(rgb_row, lms);
    let f1 = dot(rgb_row, first_lms);
    let f2 = dot(rgb_row, second_lms);
    return saturation - f * f1 / (f1 * f1 - 0.5 * f * f2);
}

fn reinhard_gamut_connected_saturation(hue: vec2<f32>, saturation: f32) -> f32 {
    let blue_notch_axis = vec2<f32>(-0.10362546, -0.99461639);
    let alignment = max(dot(hue, blue_notch_axis), 0.0);
    let alignment2 = alignment * alignment;
    let alignment4 = alignment2 * alignment2;
    let alignment8 = alignment4 * alignment4;
    let alignment16 = alignment8 * alignment8;
    let alignment32 = alignment16 * alignment16;
    let alignment64 = alignment32 * alignment32;
    let alignment128 = alignment64 * alignment64;
    let alignment256 = alignment128 * alignment128;
    return min(saturation, 0.57 + (1.0 - alignment256));
}

fn reinhard_gamut_cusp_lightness(saturation: f32, direction: vec3<f32>) -> f32 {
    let roots = vec3<f32>(1.0) + saturation * direction;
    let lms = roots * roots * roots;
    let rgb = vec3<f32>(
        dot(REINHARD_GAMUT_RED_ROW, lms),
        dot(REINHARD_GAMUT_GREEN_ROW, lms),
        dot(REINHARD_GAMUT_BLUE_ROW, lms),
    );
    return pow(1.0 / max(rgb.r, max(rgb.g, rgb.b)), 1.0 / 3.0);
}

fn reinhard_gamut_refine_upper_chroma(
    chroma: f32,
    lightness: f32,
    direction: vec3<f32>,
) -> f32 {
    let roots = vec3<f32>(lightness) + chroma * direction;
    let lms = roots * roots * roots;
    let first_lms = 3.0 * direction * roots * roots;
    let second_lms = 6.0 * direction * direction * roots;
    let rgb = vec3<f32>(
        dot(REINHARD_GAMUT_RED_ROW, lms),
        dot(REINHARD_GAMUT_GREEN_ROW, lms),
        dot(REINHARD_GAMUT_BLUE_ROW, lms),
    );
    let first_rgb = vec3<f32>(
        dot(REINHARD_GAMUT_RED_ROW, first_lms),
        dot(REINHARD_GAMUT_GREEN_ROW, first_lms),
        dot(REINHARD_GAMUT_BLUE_ROW, first_lms),
    );
    let second_rgb = vec3<f32>(
        dot(REINHARD_GAMUT_RED_ROW, second_lms),
        dot(REINHARD_GAMUT_GREEN_ROW, second_lms),
        dot(REINHARD_GAMUT_BLUE_ROW, second_lms),
    );
    let f = rgb - vec3<f32>(1.0);
    let denominator = first_rgb * first_rgb - 0.5 * f * second_rgb;
    let reciprocal_step = first_rgb / denominator;
    var step = -f * reciprocal_step;
    step = select(vec3<f32>(1.0e20), step, reciprocal_step >= vec3<f32>(0.0));
    return chroma + min(step.r, min(step.g, step.b));
}

fn reinhard_gamut_soft_min(value: f32, limit: f32, power: f32) -> f32 {
    if (value <= 0.0 || limit <= 0.0) {
        return 0.0;
    }
    let lower = min(value, limit);
    let higher = max(value, limit);
    let ratio = lower / higher;
    return lower * pow(1.0 + pow(ratio, power), -1.0 / power);
}

fn reinhard_gamut_soft_min4(value: f32, limit: f32) -> f32 {
    if (value <= 0.0 || limit <= 0.0) {
        return 0.0;
    }
    let lower = min(value, limit);
    let higher = max(value, limit);
    let ratio = lower / higher;
    let ratio2 = ratio * ratio;
    let root = sqrt(1.0 + ratio2 * ratio2);
    return lower * inverseSqrt(root);
}

fn reinhard_gamut_saturation_cap(
    lightness: f32,
    maximum_saturation: f32,
    direction: vec3<f32>,
) -> f32 {
    if (lightness <= 0.0) {
        return maximum_saturation;
    }
    if (lightness >= 1.0) {
        return 0.0;
    }

    let cusp_lightness = reinhard_gamut_cusp_lightness(maximum_saturation, direction);
    let black_chroma = lightness * maximum_saturation;
    var white_chroma = cusp_lightness * maximum_saturation
        * (1.0 - lightness) / (1.0 - cusp_lightness);
    white_chroma = reinhard_gamut_refine_upper_chroma(white_chroma, lightness, direction);

    let t = clamp((lightness - cusp_lightness) / (1.0 - cusp_lightness), 0.0, 1.0);
    let shoulder = t * (1.0 - t);
    white_chroma *= 1.0 - 0.0035 * 16.0 * shoulder * shoulder;

    let rounded_chroma = reinhard_gamut_soft_min4(black_chroma, white_chroma);
    return max(rounded_chroma / lightness, 0.0);
}

fn reinhard_gamut_chroma_retention(lightness: f32) -> f32 {
    let lightness2 = lightness * lightness;
    let lightness4 = lightness2 * lightness2;
    let lightness8 = lightness4 * lightness4;
    return 1.0 - lightness8 * lightness4;
}

fn reinhard_gamut_rounding_power(lightness: f32) -> f32 {
    let endpoint_distance = lightness * (1.0 - lightness);
    return 32.0 - 256.0 * endpoint_distance * endpoint_distance;
}

fn reinhard_gamut_map(color: vec3<f32>, overexposure: f32) -> vec3<f32> {
    let oklab = reinhard_gamut_rgb_to_oklab(max(color, vec3<f32>(0.0)));
    if (oklab.x <= 0.0) {
        return vec3<f32>(0.0);
    }

    let output_lightness = reinhard_gamut_map_lightness(oklab.x, overexposure);
    let input_chroma = length(oklab.yz);
    if (input_chroma <= 1.0e-8) {
        return REINHARD_GAMUT_RGB_HEADROOM
            * reinhard_gamut_oklab_to_rgb(vec3<f32>(output_lightness, 0.0, 0.0));
    }

    let hue = oklab.yz / input_chroma;
    let direction = reinhard_gamut_root_direction(hue);
    let input_saturation = input_chroma / oklab.x;
    let maximum_saturation = reinhard_gamut_connected_saturation(
        hue,
        reinhard_gamut_max_saturation(hue, direction),
    );
    let desired_saturation = input_saturation
        * reinhard_gamut_chroma_retention(output_lightness);
    let saturation_cap = reinhard_gamut_saturation_cap(
        output_lightness,
        maximum_saturation,
        direction,
    );
    let output_saturation = reinhard_gamut_soft_min(
        desired_saturation,
        saturation_cap,
        reinhard_gamut_rounding_power(output_lightness),
    );
    return REINHARD_GAMUT_RGB_HEADROOM * reinhard_gamut_oklab_to_rgb(vec3<f32>(
        output_lightness,
        output_lightness * output_saturation * hue,
    ));
}
