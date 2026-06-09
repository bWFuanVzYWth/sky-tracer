const float PI = 3.141592653589793;
const float SQRT_PI = 1.772453850905516;
const float TWO_INV_SQRT_PI = 1.1283791670955126;
const float INV_PI = 0.31830988618379067;
const float INV_4PI = 0.07957747154594767;
const float RAYLEIGH_PHASE_SCALE = 0.05968310365946075;

// Shadertoy-style scene constants. Fixed physical and color calibration knobs
// live here so this stays a mostly self-contained single-file port.
const float PLANET_BOTTOM_RADIUS_KM = 6360.0;
const float ATMOSPHERE_TOP_RADIUS_KM = 6460.0;
const float OBSERVER_ALTITUDE_KM = 0.2;
const float SUN_ELEVATION_DEG = 0.0;
const float SUN_AZIMUTH_DEG = 0.0;
const float SUN_MOUSE_MIN_ELEVATION_DEG = -10.0;
const float SUN_MOUSE_MAX_ELEVATION_DEG = 90.0;
const float CAMERA_YAW_DEG = 0.0;
const float CAMERA_PITCH_DEG = 0.0;
const float CAMERA_FOV_Y_DEG = 65.0;
const float SHADERTOY_EXPOSURE = 0.015;

const float SUN_ANGULAR_RADIUS_RAD = 0.00471;
const float SUN_COS_ANGULAR_RADIUS = 0.99998891;
const vec3 SUN_IRRADIANCE_REC2020_W_M2 = vec3(205.0, 205.0, 205.0);
const vec4 SUN_SPECTRAL_IRRADIANCE = vec4(1.679, 1.828, 1.986, 1.307);
const vec4 RAYLEIGH_SCATTERING_BASE_KM_INV = vec4(6.605e-3, 1.067e-2, 1.842e-2, 3.156e-2);
const vec4 MIE_SCATTERING_UNSCALED_KM_INV = vec4(1.25e-3, 1.69e-3, 2.53e-3, 3.40e-3);
const float MIE_CONCENTRATION_SCALE = 10.0;
const float MIE_EXTINCTION_SCALE = 1.11;
// 630/560/490/430 nm ozone peak absorption for a 0..44 km triangular layer
// centered at 22 km, matched to the vertical ozone column in data/atmosphere_profile.csv.
const vec4 OZONE_ABSORPTION_PEAK_KM_INV = vec4(1.37334e-3, 1.39023e-3, 5.15003e-4, 1.68431e-5);

// Rayleigh and Mie use a radial quadratic exponential density:
//   rho(r) = exp(-(r^2 - Rg^2) / (2 * Rg * H)).
// H is the near-ground equivalent scale height, fitted from the offline data
// as vertical column / surface density rather than a textbook exponential.
const float RAYLEIGH_EQUIVALENT_HEIGHT_KM = 8.4675;
const float MIE_EQUIVALENT_HEIGHT_KM = 1.6670;
const float MIE_HACK_PHASE_E = 3500.0;
const float OZONE_CENTER_ALTITUDE_KM = 22.0;
const float OZONE_HALF_WIDTH_KM = 22.0;
const float GROUND_ALBEDO = 0.18;

// Numerical and quality controls.
const float SEGMENT_MIDPOINT_U = 0.5;
const float PLANET_RADIUS_EPS_KM = 1.0e-3;
const float MIN_SCALE_HEIGHT_KM = 1.0e-4;
const float DISTANCE_EPS_KM = 1.0e-6;
const float GEOMETRY_EPS = 1.0e-8;
const float LOG_ARG_EPS = 1.0e-20;
const float NO_INTERSECTION = -1.0;
const float OPAQUE_COLUMN = 1.0e8;
const float SUN_DIRECTIONAL_COS_RADIUS = 0.999999;
const float ERFCX_MAX_X = 100.0;
const float ERFCX_RATIONAL_SWITCH_X = 8.0;
const float ERFI_SERIES_SWITCH_X = 2.5;
const int ERFI_SERIES_TERMS = 12;
const float OPTICAL_DEPTH_CLAMP = 80.0;
const float TRANSMITTANCE_LINEAR_TAU_EPS = 1.0e-3;
const float TRANSMITTANCE_QUADRATIC_Q_EPS = 1.0e-3;
const float TRANSMITTANCE_QUADRATIC_U_MIN = 0.05;
const float TRANSMITTANCE_QUADRATIC_U_MAX = 0.95;
const float COLUMN_RATIO_EPS = 1.0e-6;
const float GROUND_SKY_RAYLEIGH_PHASE_MU_SCALE = 0.5;

float bottomRadiusKm() {
    return PLANET_BOTTOM_RADIUS_KM;
}

float topRadiusKm() {
    return ATMOSPHERE_TOP_RADIUS_KM;
}

float eyeRadiusKm() {
    return clamp(bottomRadiusKm() + OBSERVER_ALTITUDE_KM, bottomRadiusKm() + PLANET_RADIUS_EPS_KM, topRadiusKm() - PLANET_RADIUS_EPS_KM);
}

vec2 defaultSunUv() {
    float azimuthU = clamp((SUN_AZIMUTH_DEG + 180.0) / 360.0, 0.0, 1.0);
    float elevationU = clamp((SUN_ELEVATION_DEG - SUN_MOUSE_MIN_ELEVATION_DEG) / (SUN_MOUSE_MAX_ELEVATION_DEG - SUN_MOUSE_MIN_ELEVATION_DEG), 0.0, 1.0);
    return vec2(azimuthU, elevationU);
}

vec3 toSunDir() {
    bool hasMouse = dot(abs(iMouse.zw), vec2(1.0)) > 0.0;
    vec2 sunUv = hasMouse ? clamp(iMouse.xy / max(iResolution.xy, vec2(1.0)), vec2(0.0), vec2(1.0)) : defaultSunUv();
    float azimuth = radians(mix(-180.0, 180.0, sunUv.x));
    float elevation = radians(mix(SUN_MOUSE_MIN_ELEVATION_DEG, SUN_MOUSE_MAX_ELEVATION_DEG, sunUv.y));
    float cosE = cos(elevation);
    return normalize(vec3(sin(azimuth) * cosE, sin(elevation), cos(azimuth) * cosE));
}

vec4 mieScatteringBaseKmInv() {
    return MIE_SCATTERING_UNSCALED_KM_INV * MIE_CONCENTRATION_SCALE;
}

vec4 mieExtinctionBaseKmInv() {
    return mieScatteringBaseKmInv() * MIE_EXTINCTION_SCALE;
}

float erfApprox(float xIn) {
    float s = xIn >= 0.0 ? 1.0 : -1.0;
    float x = abs(xIn);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-x * x));
}

float erfiApprox(float xIn) {
    float s = xIn >= 0.0 ? 1.0 : -1.0;
    float x = abs(xIn);
    if(x < ERFI_SERIES_SWITCH_X) {
        float xx = x * x;
        float term = x;
        float sum = x;
        for(int i = 1; i <= ERFI_SERIES_TERMS; ++i) {
            float n = float(i);
            term *= xx * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
            sum += term;
        }
        return s * TWO_INV_SQRT_PI * sum;
    }

    float invX2 = 1.0 / max(x * x, COLUMN_RATIO_EPS);
    float series = 1.0 + 0.5 * invX2 + 0.75 * invX2 * invX2 + 1.875 * invX2 * invX2 * invX2 + 6.5625 * invX2 * invX2 * invX2 * invX2;
    return s * exp(min(x * x, OPTICAL_DEPTH_CLAMP)) * series / (SQRT_PI * max(x, MIN_SCALE_HEIGHT_KM));
}

// Source: Vasylyev et al. 2021, "Approximate Chapman function for high zenith angles",
// Earth, Planets and Space, Eq. 37. This only reuses the stable erfcx fit.
// https://doi.org/10.1186/s40623-021-01435-y
float erfcxVasylyev(float xIn) {
    float x = clamp(xIn, 0.0, ERFCX_MAX_X);
    if(x <= ERFCX_RATIONAL_SWITCH_X) {
        return (1.0606963 + 0.55643831 * x) / (1.0619896 + 1.7245609 * x + x * x);
    }
    return 0.56498823 / (0.06651874 + x);
}

float raySphereIntersection(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    if(c > 0.0 && b > 0.0) {
        return NO_INTERSECTION;
    }
    float d = b * b - c;
    if(d < 0.0) {
        return NO_INTERSECTION;
    }
    if(d > b * b) {
        return -b + sqrt(d);
    }
    return -b - sqrt(d);
}

float eyeGroundIntersection(vec3 dir) {
    float r = eyeRadiusKm();
    float ground = bottomRadiusKm();
    float horizonMu2 = max(1.0 - (ground * ground) / max(r * r, GEOMETRY_EPS), 0.0);
    float mu = normalize(dir).y;
    if(mu >= 0.0 || mu * mu < horizonMu2) {
        return NO_INTERSECTION;
    }
    return -r * mu - r * sqrt(max(mu * mu - horizonMu2, 0.0));
}

float rayleighPhase(float mu) {
    return RAYLEIGH_PHASE_SCALE * (1.0 + mu * mu);
}

float opacLikeMiePhaseHack(float mu) {
    // Empirical Alpha-Piscium/Jessie Klein-Nishina-style phase. This is not
    // physically correct, but it is cheap, normalized, and visually close to
    // OPAC's strong forward lobe.
    float e = MIE_HACK_PHASE_E;
    return e / (2.0 * PI * (e * (1.0 - clamp(mu, -1.0, 1.0)) + 1.0) * log(2.0 * e + 1.0));
}

float raySphereExitDistance(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float d = b * b - c;
    if(d < 0.0) {
        return NO_INTERSECTION;
    }
    float root = sqrt(d);
    float t0 = -b - root;
    float t1 = -b + root;
    if(t1 >= 0.0) {
        return t1;
    }
    if(t0 >= 0.0) {
        return t0;
    }
    return NO_INTERSECTION;
}

float radialQuadraticColumnPositiveSide(float c, float b, float distance, float a) {
    float sqrtA = sqrt(a);
    float x0 = max(b, 0.0) * sqrtA;
    float x1 = x0 + distance * sqrtA;
    float attenuation = exp(clamp(-(x1 * x1 - x0 * x0), -OPTICAL_DEPTH_CLAMP, 0.0));
    float erfcxDelta = max(erfcxVasylyev(x0) - attenuation * erfcxVasylyev(x1), 0.0);
    float densityBase = exp(clamp(-a * max(c, 0.0), -OPTICAL_DEPTH_CLAMP, 0.0));
    return densityBase * SQRT_PI * erfcxDelta / (2.0 * sqrtA);
}

float radialQuadraticDensityColumnSegment(vec3 posKm, vec3 dir, float distanceKm, float scaleHeightKm) {
    float distance = max(distanceKm, 0.0);
    if(distance <= DISTANCE_EPS_KM) {
        return 0.0;
    }

    vec3 d = normalize(dir);
    float groundRadius = bottomRadiusKm();
    float h = max(scaleHeightKm, MIN_SCALE_HEIGHT_KM);
    float a = 1.0 / (2.0 * groundRadius * h);
    float c = max(dot(posKm, posKm) - groundRadius * groundRadius, 0.0);
    float b = dot(posKm, d);

    if(b >= 0.0) {
        return radialQuadraticColumnPositiveSide(c, b, distance, a);
    }

    float bEnd = b + distance;
    if(bEnd <= 0.0) {
        vec3 endPos = posKm + d * distance;
        float endC = max(dot(endPos, endPos) - groundRadius * groundRadius, 0.0);
        return radialQuadraticColumnPositiveSide(endC, -bEnd, distance, a);
    }

    float sqrtA = sqrt(a);
    float x0 = b * sqrtA;
    float x1 = bEnd * sqrtA;
    float tangentExcessRadius2 = max(c - b * b, 0.0);
    float tangentDensity = exp(clamp(-a * tangentExcessRadius2, -OPTICAL_DEPTH_CLAMP, 0.0));
    return max(tangentDensity * SQRT_PI * (erfApprox(x1) - erfApprox(x0)) / (2.0 * sqrtA), 0.0);
}

float columnToTop(vec3 posKm, vec3 dir, float scaleHeightKm) {
    vec3 d = normalize(dir);
    if(raySphereIntersection(posKm, d, bottomRadiusKm()) >= 0.0) {
        return OPAQUE_COLUMN;
    }
    float tTop = raySphereExitDistance(posKm, d, topRadiusKm());
    if(tTop < 0.0) {
        return 0.0;
    }
    return radialQuadraticDensityColumnSegment(posKm, d, tTop, scaleHeightKm);
}

float radialRampPrimitive(vec3 posKm, vec3 dir, float s, float thresholdRadiusKm) {
    float b = dot(posKm, dir);
    float p2 = max(dot(posKm, posKm) - b * b, GEOMETRY_EPS);
    float p = sqrt(p2);
    float x = s + b;
    float radius = sqrt(x * x + p2);
    float sqrtIntegral = 0.5 * (x * radius + p2 * log(max((x + radius) / p, LOG_ARG_EPS)));
    return sqrtIntegral - thresholdRadiusKm * x;
}

float radialRampInterval(vec3 posKm, vec3 dir, float lo, float hi, float thresholdRadiusKm) {
    if(hi <= lo) {
        return 0.0;
    }

    float b = dot(posKm, dir);
    float p2 = max(dot(posKm, posKm) - b * b, 0.0);
    float threshold2 = thresholdRadiusKm * thresholdRadiusKm;
    float mass = 0.0;
    if(p2 >= threshold2) {
        mass = radialRampPrimitive(posKm, dir, hi, thresholdRadiusKm) - radialRampPrimitive(posKm, dir, lo, thresholdRadiusKm);
    } else {
        float root = sqrt(max(threshold2 - p2, 0.0));
        float enter = -b - root;
        float exit = -b + root;
        float hi0 = min(hi, enter);
        if(hi0 > lo) {
            mass += radialRampPrimitive(posKm, dir, hi0, thresholdRadiusKm) - radialRampPrimitive(posKm, dir, lo, thresholdRadiusKm);
        }
        float lo1 = max(lo, exit);
        if(hi > lo1) {
            mass += radialRampPrimitive(posKm, dir, hi, thresholdRadiusKm) - radialRampPrimitive(posKm, dir, lo1, thresholdRadiusKm);
        }
    }
    return max(mass, 0.0);
}

float ozoneTriangleColumnSegment(vec3 posKm, vec3 dir, float tMaxKm) {
    if(tMaxKm <= 0.0) {
        return 0.0;
    }

    vec3 d = normalize(dir);
    float bottom = bottomRadiusKm();
    float r0 = bottom + max(OZONE_CENTER_ALTITUDE_KM - OZONE_HALF_WIDTH_KM, 0.0);
    float r1 = bottom + OZONE_CENTER_ALTITUDE_KM;
    float r2 = bottom + OZONE_CENTER_ALTITUDE_KM + OZONE_HALF_WIDTH_KM;

    float b = dot(posKm, d);
    float p2 = max(dot(posKm, posKm) - b * b, 0.0);
    float disc = r2 * r2 - p2;
    if(disc <= 0.0) {
        return 0.0;
    }

    float root = sqrt(disc);
    float lo = max(0.0, -b - root);
    float hi = min(tMaxKm, -b + root);
    if(hi <= lo) {
        return 0.0;
    }

    float mass = radialRampInterval(posKm, d, lo, hi, r0) - 2.0 * radialRampInterval(posKm, d, lo, hi, r1) + radialRampInterval(posKm, d, lo, hi, r2);
    return max(mass / max(OZONE_HALF_WIDTH_KM, MIN_SCALE_HEIGHT_KM), 0.0);
}

float ozoneColumnToTop(vec3 posKm, vec3 dir) {
    vec3 d = normalize(dir);
    if(raySphereIntersection(posKm, d, bottomRadiusKm()) >= 0.0) {
        return OPAQUE_COLUMN;
    }
    float tTop = raySphereExitDistance(posKm, d, topRadiusKm());
    if(tTop < 0.0) {
        return 0.0;
    }
    return ozoneTriangleColumnSegment(posKm, d, tTop);
}

vec4 opticalDepthFromColumns(float rayleighCol, float mieCol, float ozoneCol) {
    return RAYLEIGH_SCATTERING_BASE_KM_INV * rayleighCol + mieExtinctionBaseKmInv() * mieCol + OZONE_ABSORPTION_PEAK_KM_INV * ozoneCol;
}

vec4 opticalDepthSegment(vec3 posKm, vec3 dir, float distanceKm) {
    vec3 d = normalize(dir);
    float distance = max(distanceKm, 0.0);
    float rayleighCol = radialQuadraticDensityColumnSegment(posKm, d, distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float mieCol = radialQuadraticDensityColumnSegment(posKm, d, distance, MIE_EQUIVALENT_HEIGHT_KM);
    float ozoneCol = ozoneTriangleColumnSegment(posKm, d, distance);
    return opticalDepthFromColumns(rayleighCol, mieCol, ozoneCol);
}

float averageTransmittanceLinearScalar(float tau0, float tau1) {
    float d = tau1 - tau0;
    if(abs(d) < TRANSMITTANCE_LINEAR_TAU_EPS) {
        return exp(-0.5 * (tau0 + tau1));
    }
    return (exp(-tau0) - exp(-tau1)) / d;
}

float averageTransmittanceQuadraticScalar(float tau0, float tauMid, float tau1, float uMidIn) {
    if(tauMid > OPTICAL_DEPTH_CLAMP) {
        return 0.0;
    }
    if(tau0 > OPTICAL_DEPTH_CLAMP || tau1 > OPTICAL_DEPTH_CLAMP) {
        return averageTransmittanceLinearScalar(tau0, tau1);
    }
    if(uMidIn <= TRANSMITTANCE_QUADRATIC_U_MIN || uMidIn >= TRANSMITTANCE_QUADRATIC_U_MAX) {
        return averageTransmittanceLinearScalar(tau0, tau1);
    }

    float uMid = clamp(uMidIn, TRANSMITTANCE_QUADRATIC_U_MIN, TRANSMITTANCE_QUADRATIC_U_MAX);
    float d = tau1 - tau0;
    float q = (tauMid - (tau0 + d * uMid)) / max(uMid * (1.0 - uMid), MIN_SCALE_HEIGHT_KM);
    if(abs(q) < TRANSMITTANCE_QUADRATIC_Q_EPS) {
        return averageTransmittanceLinearScalar(tau0, tau1);
    }

    float b = d + q;
    if(q > 0.0) {
        float sqrtQ = sqrt(q);
        float x0 = -b / (2.0 * sqrtQ);
        float x1 = sqrtQ + x0;
        float scale = exp(clamp(-tau0 - b * b / (4.0 * q), -OPTICAL_DEPTH_CLAMP, OPTICAL_DEPTH_CLAMP));
        float integral = scale * SQRT_PI * (erfiApprox(x1) - erfiApprox(x0)) / (2.0 * sqrtQ);
        return max(integral, 0.0);
    }

    float p = -q;
    float sqrtP = sqrt(p);
    float x0 = b / (2.0 * sqrtP);
    float x1 = sqrtP + x0;
    float scale = exp(clamp(-tau0 + b * b / (4.0 * p), -OPTICAL_DEPTH_CLAMP, OPTICAL_DEPTH_CLAMP));
    float integral = scale * SQRT_PI * (erfApprox(x1) - erfApprox(x0)) / (2.0 * sqrtP);
    return max(integral, 0.0);
}

vec4 averageTransmittanceQuadratic(vec4 tau0, vec4 tauMid, vec4 tau1, float uMid) {
    return vec4(averageTransmittanceQuadraticScalar(tau0.x, tauMid.x, tau1.x, uMid), averageTransmittanceQuadraticScalar(tau0.y, tauMid.y, tau1.y, uMid), averageTransmittanceQuadraticScalar(tau0.z, tauMid.z, tau1.z, uMid), averageTransmittanceQuadraticScalar(tau0.w, tauMid.w, tau1.w, uMid));
}

vec4 groundBounceTransfer(vec3 posKm, vec3 sunDir) {
    float radius = max(length(posKm), bottomRadiusKm() + PLANET_RADIUS_EPS_KM);
    vec3 up = posKm / radius;
    float sunCos = max(dot(up, sunDir), 0.0);
    if(sunCos <= 0.0) {
        return vec4(0.0);
    }

    float bottom = bottomRadiusKm();
    vec3 groundPos = up * (bottom + PLANET_RADIUS_EPS_KM);
    float groundToSampleDistance = max(radius - (bottom + PLANET_RADIUS_EPS_KM), 0.0);
    float horizon = sqrt(max(radius * radius - bottom * bottom, 0.0));
    float planetSolidAngle = 2.0 * PI * (1.0 - horizon / radius);

    vec4 sunGroundTau = opticalDepthFromColumns(columnToTop(groundPos, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM), columnToTop(groundPos, sunDir, MIE_EQUIVALENT_HEIGHT_KM), ozoneColumnToTop(groundPos, sunDir));
    float groundToSampleR = radialQuadraticDensityColumnSegment(groundPos, up, groundToSampleDistance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float groundToSampleM = radialQuadraticDensityColumnSegment(groundPos, up, groundToSampleDistance, MIE_EQUIVALENT_HEIGHT_KM);
    float groundToSampleO = ozoneTriangleColumnSegment(groundPos, up, groundToSampleDistance);
    vec4 groundToSampleTau = opticalDepthFromColumns(groundToSampleR, groundToSampleM, groundToSampleO);

    return vec4(INV_4PI * GROUND_ALBEDO * INV_PI * planetSolidAngle * sunCos) * exp(-(sunGroundTau + groundToSampleTau));
}

vec4 groundDirectIrradianceTransfer(vec3 groundPosKm, vec3 normal, vec3 sunDir) {
    float sunCos = max(dot(normal, sunDir), 0.0);
    if(sunCos <= 0.0) {
        return vec4(0.0);
    }
    vec4 sunTau = opticalDepthFromColumns(columnToTop(groundPosKm, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM), columnToTop(groundPosKm, sunDir, MIE_EQUIVALENT_HEIGHT_KM), ozoneColumnToTop(groundPosKm, sunDir));
    return vec4(sunCos) * exp(-sunTau);
}

vec4 groundSkyIrradianceTransferApprox(vec3 groundPosKm, vec3 normal, vec3 sunDir) {
    float sunCos = max(dot(normal, sunDir), 0.0);
    if(sunCos <= 0.0) {
        return vec4(0.0);
    }

    float tTop = raySphereIntersection(groundPosKm, normal, topRadiusKm());
    if(tTop <= 0.0) {
        return vec4(0.0);
    }

    vec3 pMid = groundPosKm + normal * (SEGMENT_MIDPOINT_U * tTop);
    vec3 pEnd = groundPosKm + normal * tTop;
    float viewColR = columnToTop(groundPosKm, normal, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColM = columnToTop(groundPosKm, normal, MIE_EQUIVALENT_HEIGHT_KM);
    float viewColMidR = radialQuadraticDensityColumnSegment(groundPosKm, normal, SEGMENT_MIDPOINT_U * tTop, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColMidM = radialQuadraticDensityColumnSegment(groundPosKm, normal, SEGMENT_MIDPOINT_U * tTop, MIE_EQUIVALENT_HEIGHT_KM);
    float viewColO = ozoneTriangleColumnSegment(groundPosKm, normal, tTop);
    float viewColMidO = ozoneTriangleColumnSegment(groundPosKm, normal, SEGMENT_MIDPOINT_U * tTop);

    float sunCol0R = columnToTop(groundPosKm, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol0M = columnToTop(groundPosKm, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol0O = ozoneColumnToTop(groundPosKm, sunDir);
    float sunColMidR = columnToTop(pMid, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunColMidM = columnToTop(pMid, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunColMidO = ozoneColumnToTop(pMid, sunDir);
    float sunCol1R = columnToTop(pEnd, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol1M = columnToTop(pEnd, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol1O = ozoneColumnToTop(pEnd, sunDir);

    vec4 tauS0 = opticalDepthFromColumns(sunCol0R, sunCol0M, sunCol0O);
    vec4 tauMid = opticalDepthFromColumns(sunColMidR, sunColMidM, sunColMidO) + opticalDepthFromColumns(viewColMidR, viewColMidM, viewColMidO);
    vec4 tauS1V = opticalDepthFromColumns(sunCol1R, sunCol1M, sunCol1O) + opticalDepthFromColumns(viewColR, viewColM, viewColO);
    vec4 avgTR = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidR / max(viewColR, COLUMN_RATIO_EPS));
    vec4 avgTM = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidM / max(viewColM, COLUMN_RATIO_EPS));

    vec4 rayleigh = RAYLEIGH_SCATTERING_BASE_KM_INV * viewColR * rayleighPhase(GROUND_SKY_RAYLEIGH_PHASE_MU_SCALE * sunCos) * avgTR;
    vec4 mie = mieScatteringBaseKmInv() * viewColM * INV_4PI * avgTM;
    return PI * (rayleigh + mie);
}

vec4 viewSegmentScatterToGround(vec3 origin, vec3 groundPosKm, vec3 sunDir) {
    vec3 rayDir = normalize(groundPosKm - origin);
    float distance = length(groundPosKm - origin);
    if(distance <= DISTANCE_EPS_KM) {
        return vec4(0.0);
    }

    vec3 pMid = origin + rayDir * (SEGMENT_MIDPOINT_U * distance);
    float viewColR = radialQuadraticDensityColumnSegment(origin, rayDir, distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColM = radialQuadraticDensityColumnSegment(origin, rayDir, distance, MIE_EQUIVALENT_HEIGHT_KM);
    float viewColMidR = radialQuadraticDensityColumnSegment(origin, rayDir, SEGMENT_MIDPOINT_U * distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColMidM = radialQuadraticDensityColumnSegment(origin, rayDir, SEGMENT_MIDPOINT_U * distance, MIE_EQUIVALENT_HEIGHT_KM);
    float viewColO = ozoneTriangleColumnSegment(origin, rayDir, distance);
    float viewColMidO = ozoneTriangleColumnSegment(origin, rayDir, SEGMENT_MIDPOINT_U * distance);

    float sunCol0R = columnToTop(origin, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol0M = columnToTop(origin, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol0O = ozoneColumnToTop(origin, sunDir);
    float sunColMidR = columnToTop(pMid, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunColMidM = columnToTop(pMid, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunColMidO = ozoneColumnToTop(pMid, sunDir);
    float sunCol1R = columnToTop(groundPosKm, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol1M = columnToTop(groundPosKm, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol1O = ozoneColumnToTop(groundPosKm, sunDir);

    vec4 tauS0 = opticalDepthFromColumns(sunCol0R, sunCol0M, sunCol0O);
    vec4 tauViewMid = opticalDepthFromColumns(viewColMidR, viewColMidM, viewColMidO);
    vec4 tauView1 = opticalDepthFromColumns(viewColR, viewColM, viewColO);
    vec4 tauMid = opticalDepthFromColumns(sunColMidR, sunColMidM, sunColMidO) + tauViewMid;
    vec4 tauS1V = opticalDepthFromColumns(sunCol1R, sunCol1M, sunCol1O) + tauView1;
    vec4 avgTR = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidR / max(viewColR, COLUMN_RATIO_EPS));
    vec4 avgTM = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidM / max(viewColM, COLUMN_RATIO_EPS));

    float mu = dot(sunDir, rayDir);
    vec4 rayleighScatter = RAYLEIGH_SCATTERING_BASE_KM_INV * viewColR * rayleighPhase(mu) * avgTR;
    vec4 mieScatter = mieScatteringBaseKmInv() * viewColM * opacLikeMiePhaseHack(mu) * avgTM;
    return SUN_SPECTRAL_IRRADIANCE * (rayleighScatter + mieScatter);
}

vec3 linearRec2020FromSpectral(vec4 l) {
    return l.x * vec3(86.3182148, -0.122697755, 0.547224869) + l.y * vec3(30.0452569, 92.3535448, -8.36373448) + l.z * vec3(-1.57281544, 29.5419052, 48.5065647) + l.w * vec3(3.57535605, -9.78845357, 70.7444659);
}

vec3 whiteBalanceRec2020(vec3 rgb) {
    return rgb.x * vec3(1.01363293, 0.00103366792, 0.00115468962) + rgb.y * vec3(0.019007348, 0.974260442, -0.00255465921) + rgb.z * vec3(0.00260596377, -0.00288158643, 1.19816913);
}

vec3 whiteBalancedLinearRec2020FromSpectral(vec4 l) {
    return whiteBalanceRec2020(linearRec2020FromSpectral(l));
}

vec3 analyticGroundRadiance(vec3 origin, vec3 dir, float tGround, vec3 sunDir) {
    vec3 groundHit = origin + dir * tGround;
    vec3 normal = normalize(groundHit);
    vec3 groundPos = normal * (bottomRadiusKm() + PLANET_RADIUS_EPS_KM);
    vec3 viewToEye = normalize(origin - groundPos);
    float viewDistance = length(origin - groundPos);

    vec4 directTransfer = groundDirectIrradianceTransfer(groundPos, normal, sunDir);
    vec4 skyTransfer = groundSkyIrradianceTransferApprox(groundPos, normal, sunDir);
    vec4 viewTransmittance = exp(-opticalDepthSegment(groundPos, viewToEye, viewDistance));
    vec4 groundSpectral = SUN_SPECTRAL_IRRADIANCE * (directTransfer + skyTransfer) * vec4(GROUND_ALBEDO * INV_PI) * viewTransmittance;
    vec4 viewScatterSpectral = viewSegmentScatterToGround(origin, groundPos, sunDir);
    return max(whiteBalancedLinearRec2020FromSpectral(groundSpectral + viewScatterSpectral), vec3(0.0));
}

vec3 spectralSunTransmittanceToRec2020(vec4 transmittance) {
    vec3 clearSun = max(whiteBalancedLinearRec2020FromSpectral(SUN_SPECTRAL_IRRADIANCE), vec3(COLUMN_RATIO_EPS));
    vec3 attenuatedSun = max(whiteBalancedLinearRec2020FromSpectral(SUN_SPECTRAL_IRRADIANCE * transmittance), vec3(0.0));
    return clamp(attenuatedSun / clearSun, vec3(0.0), vec3(1.0));
}

vec3 sunDiskEval(vec3 direction, vec3 transmittance) {
    float cosRadius = clamp(SUN_COS_ANGULAR_RADIUS, -1.0, 1.0);
    if(cosRadius >= SUN_DIRECTIONAL_COS_RADIUS) {
        return vec3(0.0);
    }
    if(dot(normalize(direction), toSunDir()) < cosRadius) {
        return vec3(0.0);
    }
    float solidAngle = 2.0 * PI * max(1.0 - cosRadius, 0.0);
    if(solidAngle <= 0.0) {
        return vec3(0.0);
    }
    return SUN_IRRADIANCE_REC2020_W_M2 * transmittance / solidAngle;
}

vec3 analyticSkyRadiance(vec3 direction) {
    vec3 origin = vec3(0.0, eyeRadiusKm(), 0.0);
    vec3 dir = normalize(direction);
    vec3 sunDir = toSunDir();
    float tGround = eyeGroundIntersection(dir);
    if(tGround >= 0.0) {
        return analyticGroundRadiance(origin, dir, tGround, sunDir);
    }

    float tTop = raySphereIntersection(origin, dir, topRadiusKm());
    if(tTop < 0.0) {
        return vec3(0.0);
    }

    vec3 pEnd = origin + dir * tTop;
    vec3 pMid = origin + dir * (SEGMENT_MIDPOINT_U * tTop);
    float viewColR = columnToTop(origin, dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColM = columnToTop(origin, dir, MIE_EQUIVALENT_HEIGHT_KM);
    float viewColMidR = radialQuadraticDensityColumnSegment(origin, dir, SEGMENT_MIDPOINT_U * tTop, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float viewColMidM = radialQuadraticDensityColumnSegment(origin, dir, SEGMENT_MIDPOINT_U * tTop, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol0R = columnToTop(origin, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol0M = columnToTop(origin, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol0O = ozoneColumnToTop(origin, sunDir);
    float sunColMidR = columnToTop(pMid, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunColMidM = columnToTop(pMid, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunColMidO = ozoneColumnToTop(pMid, sunDir);
    float sunCol1R = columnToTop(pEnd, sunDir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    float sunCol1M = columnToTop(pEnd, sunDir, MIE_EQUIVALENT_HEIGHT_KM);
    float sunCol1O = ozoneColumnToTop(pEnd, sunDir);
    float viewColO = ozoneTriangleColumnSegment(origin, dir, tTop);
    float viewColMidO = ozoneTriangleColumnSegment(origin, dir, SEGMENT_MIDPOINT_U * tTop);

    vec4 tauS0 = opticalDepthFromColumns(sunCol0R, sunCol0M, sunCol0O);
    vec4 tauViewMid = opticalDepthFromColumns(viewColMidR, viewColMidM, viewColMidO);
    vec4 tauView1 = opticalDepthFromColumns(viewColR, viewColM, viewColO);
    vec4 tauMid = opticalDepthFromColumns(sunColMidR, sunColMidM, sunColMidO) + tauViewMid;
    vec4 tauS1V = opticalDepthFromColumns(sunCol1R, sunCol1M, sunCol1O) + tauView1;
    vec4 avgTR = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidR / max(viewColR, COLUMN_RATIO_EPS));
    vec4 avgTM = averageTransmittanceQuadratic(tauS0, tauMid, tauS1V, viewColMidM / max(viewColM, COLUMN_RATIO_EPS));
    vec4 avgViewTR = averageTransmittanceQuadratic(vec4(0.0), tauViewMid, tauView1, viewColMidR / max(viewColR, COLUMN_RATIO_EPS));
    vec4 avgViewTM = averageTransmittanceQuadratic(vec4(0.0), tauViewMid, tauView1, viewColMidM / max(viewColM, COLUMN_RATIO_EPS));

    float mu = dot(sunDir, dir);
    vec4 rayleighScatter = RAYLEIGH_SCATTERING_BASE_KM_INV * viewColR * rayleighPhase(mu) * avgTR;
    vec4 mieScatter = mieScatteringBaseKmInv() * viewColM * opacLikeMiePhaseHack(mu) * avgTM;
    vec4 groundTransfer = groundBounceTransfer(pMid, sunDir);
    vec4 groundScatter = groundTransfer * (RAYLEIGH_SCATTERING_BASE_KM_INV * viewColR * avgViewTR + mieScatteringBaseKmInv() * viewColM * avgViewTM);
    vec4 skySpectral = SUN_SPECTRAL_IRRADIANCE * (rayleighScatter + mieScatter + groundScatter);
    vec3 rgb = max(whiteBalancedLinearRec2020FromSpectral(skySpectral), vec3(0.0));

    vec3 sunT = spectralSunTransmittanceToRec2020(exp(-tauS0));
    rgb += sunDiskEval(dir, sunT);
    return rgb;
}

vec3 cameraRay(vec2 fragCoord) {
    vec2 ndc = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float yaw = radians(CAMERA_YAW_DEG);
    float pitch = radians(CAMERA_PITCH_DEG);
    pitch = clamp(pitch, radians(-85.0), radians(85.0));

    float fovTan = tan(0.5 * radians(CAMERA_FOV_Y_DEG));
    vec3 forward = normalize(vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)));
    vec3 right = normalize(vec3(cos(yaw), 0.0, -sin(yaw)));
    vec3 up = normalize(cross(forward, right));
    return normalize(forward + right * ndc.x * fovTan + up * ndc.y * fovTan);
}

vec3 displayTransform(vec3 linearRec2020) {
    // Shadertoy display helper only. analyticSkyRadiance() itself returns the
    // same white-balanced linear Rec.2020 radiance as the WGSL renderer path.
    vec3 x = max(linearRec2020 * SHADERTOY_EXPOSURE, vec3(0.0));
    x = x / (1.0 + x);
    return pow(clamp(x, 0.0, 1.0), vec3(1.0 / 2.2));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 dir = cameraRay(fragCoord);
    vec3 radiance = analyticSkyRadiance(dir);
    fragColor = vec4(displayTransform(radiance), 1.0);
}
