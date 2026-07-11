import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const SOURCE_CITIES = [
    { name: "Beijing", country: "China", lat: 39.9, lon: 116.4, pop: 21_710_000, elev: 43, tz: 480 },
    { name: "New York", country: "USA", lat: 40.7, lon: -74.0, pop: 8_419_000, elev: 10, tz: -240 },
    { name: "London", country: "UK", lat: 51.5, lon: -0.1, pop: 8_982_000, elev: 11, tz: 0 },
    { name: "Tokyo", country: "Japan", lat: 35.7, lon: 139.7, pop: 13_960_000, elev: 40, tz: 540 },
    { name: "Shanghai", country: "China", lat: 31.2, lon: 121.5, pop: 24_180_000, elev: 4, tz: 480 },
    { name: "Moscow", country: "Russia", lat: 55.8, lon: 37.6, pop: 11_920_000, elev: 156, tz: 180 },
    { name: "Sydney", country: "Australia", lat: -33.9, lon: 151.2, pop: 5_312_000, elev: 58, tz: 600 },
    { name: "Cape Town", country: "South Africa", lat: -33.9, lon: 18.4, pop: 4_337_000, elev: 5, tz: 120 },
    { name: "Rio de Janeiro", country: "Brazil", lat: -22.9, lon: -43.2, pop: 6_760_000, elev: 2, tz: -180 },
    { name: "Dubai", country: "UAE", lat: 25.2, lon: 55.3, pop: 3_395_000, elev: 0, tz: 240 },
    { name: "Delhi", country: "India", lat: 28.6, lon: 77.2, pop: 16_790_000, elev: 216, tz: 330 },
    { name: "Cairo", country: "Egypt", lat: 30.0, lon: 31.2, pop: 9_500_000, elev: 23, tz: 120 },
];

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const byId = (id) => document.getElementById(id);

function requireBytes(view, offset, length) {
    if (offset < 0 || length < 0 || offset + length > view.byteLength) {
        throw new RangeError(`ROM read outside buffer at ${offset} for ${length} bytes`);
    }
}

function encodeROM(cities) {
    const stringList = Array.from(new Set(cities.flatMap((city) => [city.name, city.country])));
    const stringMap = new Map(stringList.map((value, index) => [value, index]));
    const encodedStrings = stringList.map((value) => {
        const bytes = textEncoder.encode(value);
        if (bytes.length > 255) throw new RangeError(`ROM string exceeds 255 bytes: ${value}`);
        return bytes;
    });

    const headerSize = 14;
    const stringTableSize = encodedStrings.reduce((total, bytes) => total + 1 + bytes.length, 0);
    const cityRecordSize = 20;
    const buffer = new ArrayBuffer(headerSize + stringTableSize + cities.length * cityRecordSize);
    const view = new DataView(buffer);
    let offset = 0;

    view.setUint32(offset, 0x45525448, false); offset += 4;
    view.setUint8(offset, 0x02); offset += 1;
    view.setUint8(offset, 0x03); offset += 1;
    view.setUint16(offset, cities.length, false); offset += 2;
    view.setUint16(offset, stringList.length, false); offset += 2;
    view.setUint32(offset, headerSize, false); offset += 4;

    encodedStrings.forEach((bytes) => {
        view.setUint8(offset, bytes.length); offset += 1;
        new Uint8Array(buffer, offset, bytes.length).set(bytes);
        offset += bytes.length;
    });

    cities.forEach((city) => {
        view.setFloat32(offset, city.lat, false); offset += 4;
        view.setFloat32(offset, city.lon, false); offset += 4;
        view.setUint32(offset, city.pop, false); offset += 4;
        view.setInt16(offset, city.elev, false); offset += 2;
        view.setInt16(offset, city.tz, false); offset += 2;
        view.setUint16(offset, stringMap.get(city.name), false); offset += 2;
        view.setUint16(offset, stringMap.get(city.country), false); offset += 2;
    });

    return buffer;
}

function decodeROM(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 14) {
        throw new TypeError("ROM must be an ArrayBuffer with a complete header");
    }

    const view = new DataView(buffer);
    let offset = 0;
    requireBytes(view, offset, 14);

    const magic = view.getUint32(offset, false); offset += 4;
    if (magic !== 0x45525448) throw new Error("Invalid EarthTwin ROM magic");

    const version = view.getUint8(offset); offset += 1;
    const flags = view.getUint8(offset); offset += 1;
    const cityCount = view.getUint16(offset, false); offset += 2;
    const stringCount = view.getUint16(offset, false); offset += 2;
    const tableOffset = view.getUint32(offset, false); offset += 4;

    if (tableOffset < 14 || tableOffset >= buffer.byteLength) {
        throw new RangeError("Invalid ROM string-table offset");
    }

    offset = tableOffset;
    const strings = [];
    for (let index = 0; index < stringCount; index += 1) {
        requireBytes(view, offset, 1);
        const length = view.getUint8(offset); offset += 1;
        requireBytes(view, offset, length);
        strings.push(textDecoder.decode(new Uint8Array(buffer, offset, length)));
        offset += length;
    }

    const cities = [];
    for (let index = 0; index < cityCount; index += 1) {
        requireBytes(view, offset, 20);
        const lat = view.getFloat32(offset, false); offset += 4;
        const lon = view.getFloat32(offset, false); offset += 4;
        const pop = view.getUint32(offset, false); offset += 4;
        const elev = view.getInt16(offset, false); offset += 2;
        const tz = view.getInt16(offset, false); offset += 2;
        const nameIndex = view.getUint16(offset, false); offset += 2;
        const countryIndex = view.getUint16(offset, false); offset += 2;

        if (nameIndex >= strings.length || countryIndex >= strings.length) {
            throw new RangeError("ROM city record references an invalid string index");
        }
        cities.push({ lat, lon, pop, elev, tz, name: strings[nameIndex], country: strings[countryIndex] });
    }

    return { version, flags, cities, bufferSize: buffer.byteLength };
}

const baseROMBuffer = encodeROM(SOURCE_CITIES);
const baseROM = decodeROM(baseROMBuffer);

const meta = {
    romBuffer: baseROMBuffer,
    queryCache: new Map(),
    queryCounts: new Map(),
    totalQueries: 0,
    cacheHits: 0,
    fps: 60,
    quality: "high",
    orderedCities: [...baseROM.cities],
    hotList: [],
    refinementCounter: 0,
};

let frameCount = 0;
let lastFpsUpdate = performance.now();

function updateFPS(now) {
    frameCount += 1;
    const elapsed = now - lastFpsUpdate;
    if (elapsed < 1000) return;

    const instantaneous = frameCount * 1000 / elapsed;
    meta.fps = Math.round(meta.fps * 0.7 + instantaneous * 0.3);
    frameCount = 0;
    lastFpsUpdate = now;

    if (meta.quality === "high" && meta.fps < 38) setQuality("medium");
    else if (meta.quality === "medium" && meta.fps < 28) setQuality("low");
    else if (meta.quality === "medium" && meta.fps > 54) setQuality("high");
    else if (meta.quality === "low" && meta.fps > 42) setQuality("medium");

    updateMetaPanel();
}

function setQuality(level) {
    if (!new Set(["high", "medium", "low"]).has(level) || level === meta.quality) return;
    meta.quality = level;

    const segments = level === "high" ? 80 : level === "medium" ? 48 : 32;
    const replacement = new THREE.SphereGeometry(1, segments, segments);
    earth.geometry.dispose();
    earth.geometry = replacement;

    clouds.visible = level !== "low";
    glowMesh.visible = level !== "low";
    renderer.setPixelRatio(level === "high" ? Math.min(window.devicePixelRatio, 2) : 1);
    updateMetaPanel();
}

function refineROMLayout() {
    const sorted = [...baseROM.cities].sort((left, right) => {
        const countDifference = (meta.queryCounts.get(right.name) || 0) - (meta.queryCounts.get(left.name) || 0);
        return countDifference || left.name.localeCompare(right.name);
    });

    meta.romBuffer = encodeROM(sorted);
    meta.orderedCities = decodeROM(meta.romBuffer).cities;
    meta.hotList = sorted
        .filter((city) => (meta.queryCounts.get(city.name) || 0) > 0)
        .slice(0, 5)
        .map((city) => city.name);
    meta.queryCache.clear();
    meta.refinementCounter += 1;
    updateMetaPanel();
}

function recordCityUsage(city) {
    if (!city) return;
    const count = (meta.queryCounts.get(city.name) || 0) + 1;
    meta.queryCounts.set(city.name, count);
    if (count % 3 === 0) refineROMLayout();
}

function cacheResult(key, result, ttlMs = 300_000) {
    meta.queryCache.set(key, { result, expiresAt: Date.now() + ttlMs });
}

function readCache(key) {
    const cached = meta.queryCache.get(key);
    if (!cached) return null;
    if (cached.expiresAt <= Date.now()) {
        meta.queryCache.delete(key);
        return null;
    }
    meta.cacheHits += 1;
    return cached.result;
}

function findTargetCity(query) {
    const exact = meta.orderedCities.find((city) => city.name.toLowerCase() === query);
    if (exact) return exact;
    return meta.orderedCities.find((city) => query.includes(city.name.toLowerCase())) || null;
}

function formatUTCOffset(minutes) {
    const sign = minutes >= 0 ? "+" : "−";
    const absolute = Math.abs(minutes);
    const hours = Math.floor(absolute / 60);
    const remainder = absolute % 60;
    return `UTC${sign}${String(hours).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function comparePopulation(population, operator, threshold) {
    if (operator === "<") return population < threshold;
    if (operator === "<=") return population <= threshold;
    if (operator === ">=") return population >= threshold;
    return population > threshold;
}

function queryROM(text) {
    const query = text.toLowerCase().trim().replace(/\s+/g, " ");
    meta.totalQueries += 1;

    const cached = readCache(query);
    if (cached) {
        recordCityUsage(cached.city);
        updateMetaPanel();
        return cached;
    }

    const targetCity = findTargetCity(query);
    let result;
    let ttl = 300_000;

    const populationMatch = query.match(/(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)\s*(m|million|k|thousand)\b/i);
    if (populationMatch) {
        const operator = populationMatch[1] || ">";
        let threshold = Number.parseFloat(populationMatch[2]);
        const unit = populationMatch[3].toLowerCase();
        if (unit === "m" || unit === "million") threshold *= 1_000_000;
        else threshold *= 1_000;

        const cities = meta.orderedCities.filter((city) => comparePopulation(city.pop, operator, threshold));
        const thresholdMillions = (threshold / 1_000_000).toFixed(threshold < 10_000_000 ? 1 : 0);
        result = cities.length
            ? {
                type: "list",
                msg: `📍 Cities ${operator} ${thresholdMillions}M: ${cities.map((city) => `${city.name} (${(city.pop / 1_000_000).toFixed(1)}M)`).join(", ")}`,
                cities,
                metaTag: `ROM scan · ${cities.length} matches`,
            }
            : { type: "text", msg: `No encoded cities satisfy ${operator} ${thresholdMillions}M.`, metaTag: "ROM scan · 0 matches" };
    } else if (query.includes("time") || query.includes("clock") || query.includes("hour")) {
        if (targetCity) {
            const localTime = new Date(Date.now() + targetCity.tz * 60_000);
            const hours = String(localTime.getUTCHours()).padStart(2, "0");
            const minutes = String(localTime.getUTCMinutes()).padStart(2, "0");
            result = {
                type: "fly",
                city: targetCity,
                msg: `🕒 Local time in ${targetCity.name}: ${hours}:${minutes}`,
                metaTag: formatUTCOffset(targetCity.tz),
            };
            ttl = 10_000;
        } else {
            result = { type: "text", msg: "❓ Specify a city, for example: Time in London.", metaTag: "query incomplete" };
        }
    } else if (targetCity) {
        result = {
            type: "fly",
            city: targetCity,
            msg: `📍 ${targetCity.name}, ${targetCity.country}. Population: ${(targetCity.pop / 1_000_000).toFixed(1)}M, elevation: ${targetCity.elev}m, ${formatUTCOffset(targetCity.tz)}.`,
            metaTag: "decoded from ROM",
        };
    } else {
        result = {
            type: "text",
            msg: `No matching city was found for “${text}”. Try “Where is Tokyo?”, “Time in Delhi”, or “Show cities < 5M”.`,
            metaTag: "no ROM match",
        };
    }

    recordCityUsage(result.city);
    cacheResult(query, result, ttl);
    updateMetaPanel();
    return result;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080c);

const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.5, 3.2);

let renderer;
try {
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
} catch (error) {
    addMessage("WebGL could not initialize in this browser.", "bot", "renderer unavailable");
    throw error;
}

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputEncoding = THREE.sRGBEncoding;
document.body.prepend(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
Object.assign(labelRenderer.domElement.style, {
    position: "absolute",
    top: "0",
    left: "0",
    pointerEvents: "none",
});
document.body.prepend(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;
controls.minDistance = 1.5;
controls.maxDistance = 8;
controls.enablePan = false;

scene.add(new THREE.AmbientLight(0x404060, 0.6));

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.8);
sunLight.position.set(2, 3, 4);
scene.add(sunLight, sunLight.target);

const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
fillLight.position.set(-2, -1, -3);
scene.add(fillLight);

const earthGroup = new THREE.Group();
scene.add(earthGroup);

const textureLoader = new THREE.TextureLoader();
textureLoader.setCrossOrigin("anonymous");
const mapTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg");
const normalTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_normal_2048.jpg");
const specularTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_specular_2048.jpg");
const cloudTexture = textureLoader.load("https://threejs.org/examples/textures/planets/earth_clouds_1024.png");

const earthMaterial = new THREE.MeshPhongMaterial({
    map: mapTexture,
    normalMap: normalTexture,
    normalScale: new THREE.Vector2(1.2, 1.2),
    specularMap: specularTexture,
    specular: new THREE.Color(0x333333),
    shininess: 5,
});
const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 80, 80), earthMaterial);
earthGroup.add(earth);

const cloudMaterial = new THREE.MeshPhongMaterial({
    map: cloudTexture,
    transparent: true,
    opacity: 0.2,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    depthWrite: false,
});
const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.008, 64, 64), cloudMaterial);
earthGroup.add(clouds);

const glowMaterial = new THREE.MeshPhongMaterial({
    color: 0x4a8cff,
    transparent: true,
    opacity: 0.1,
    side: THREE.BackSide,
    depthWrite: false,
});
const glowMesh = new THREE.Mesh(new THREE.SphereGeometry(1.02, 48, 48), glowMaterial);
earthGroup.add(glowMesh);

const markerGroup = new THREE.Group();
scene.add(markerGroup);
const markers = [];

function latLonToPosition(lat, lon, radius = 1.02) {
    const phi = (90 - lat) * Math.PI / 180;
    const theta = lon * Math.PI / 180;
    return new THREE.Vector3(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta),
    );
}

function makeMarker(city) {
    const position = latLonToPosition(city.lat, city.lon);

    const dotElement = document.createElement("div");
    Object.assign(dotElement.style, {
        width: "7px",
        height: "7px",
        background: "#60f0ff",
        borderRadius: "50%",
        boxShadow: "0 0 16px #3b82f6",
        border: "1px solid rgba(255,255,255,0.4)",
        pointerEvents: "none",
    });
    const dot = new CSS2DObject(dotElement);
    dot.position.copy(position);
    markerGroup.add(dot);

    const labelElement = document.createElement("div");
    labelElement.textContent = city.name;
    Object.assign(labelElement.style, {
        color: "white",
        fontSize: "10px",
        fontWeight: "500",
        textShadow: "0 0 12px rgba(0,0,0,0.9)",
        background: "rgba(0,0,0,0.3)",
        padding: "1px 6px",
        borderRadius: "10px",
        backdropFilter: "blur(2px)",
        border: "1px solid rgba(255,255,255,0.05)",
        fontFamily: "system-ui",
        pointerEvents: "none",
        whiteSpace: "nowrap",
        transform: "translateY(-12px)",
    });
    const label = new CSS2DObject(labelElement);
    label.position.copy(position);
    markerGroup.add(label);

    markers.push({ dot, label, position, data: city });
}

baseROM.cities.forEach(makeMarker);

function dayOfYearUTC(date) {
    const start = Date.UTC(date.getUTCFullYear(), 0, 0);
    const current = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
    return Math.floor((current - start) / 86_400_000);
}

function updateSun() {
    const now = new Date();
    const hours = now.getUTCHours() + now.getUTCMinutes() / 60;
    const angle = hours / 24 * Math.PI * 2;
    const declination = 23.44 * Math.sin((dayOfYearUTC(now) - 81) * (2 * Math.PI / 365));
    sunLight.position.set(5 * Math.sin(angle), 1.2 + Math.sin(declination * Math.PI / 180) * 1.5, 5 * Math.cos(angle));
    sunLight.target.position.set(0, 0, 0);
    sunLight.target.updateMatrixWorld();
}

let flyTarget = null;
let autoRotateTimer = null;

function flyToCity(city) {
    const end = latLonToPosition(city.lat, city.lon, 1.6).normalize().multiplyScalar(2.2);
    flyTarget = { start: camera.position.clone(), end, progress: 0 };
    controls.autoRotate = false;
    window.clearTimeout(autoRotateTimer);
    autoRotateTimer = window.setTimeout(() => { controls.autoRotate = true; }, 2000);
}

function highlightMarkers(predicate) {
    markers.forEach((marker) => {
        const selected = predicate(marker.data);
        marker.dot.element.style.background = selected ? "#fcd34d" : "#60f0ff";
        marker.dot.element.style.boxShadow = selected ? "0 0 24px #fcd34d" : "0 0 16px #3b82f6";
        marker.label.element.style.fontWeight = selected ? "700" : "400";
        marker.label.element.style.color = selected ? "#fcd34d" : "white";
    });
}

function addMessage(text, sender, metaTag = "") {
    const container = byId("chat-messages");
    const message = document.createElement("div");
    message.className = `msg ${sender}`;
    message.textContent = text;

    if (metaTag) {
        const metadata = document.createElement("span");
        metadata.className = "meta-tag";
        metadata.textContent = `⚡ ${metaTag}`;
        message.appendChild(metadata);
    }

    container.appendChild(message);
    container.scrollTop = container.scrollHeight;
}

function handleChat() {
    const input = byId("chat-input");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    addMessage(text, "user");
    const result = queryROM(text);

    if (result.type === "fly") {
        flyToCity(result.city);
        highlightMarkers((city) => city.name === result.city.name);
    } else if (result.type === "list") {
        const selected = new Set(result.cities.map((city) => city.name));
        highlightMarkers((city) => selected.has(city.name));
    }

    addMessage(result.msg, "bot", result.metaTag || `cache ${meta.cacheHits}/${meta.totalQueries}`);
    input.focus();
}

function toggleChat() {
    const container = byId("chat-container");
    const collapsed = container.classList.toggle("collapsed");
    byId("chat-header").setAttribute("aria-expanded", String(!collapsed));
    byId("toggle-btn").textContent = collapsed ? "+" : "−";
    byId("toggle-btn").setAttribute("aria-label", collapsed ? "Expand chat" : "Collapse chat");
}

function toggleMeta() {
    const panel = byId("meta-panel");
    const open = panel.classList.toggle("open");
    byId("meta-toggle").setAttribute("aria-expanded", String(open));
}

function updateMetaPanel() {
    byId("m-rom-size").textContent = `${meta.romBuffer.byteLength} bytes`;
    const hitRate = meta.totalQueries ? Math.round(meta.cacheHits / meta.totalQueries * 100) : 0;
    byId("m-cache").textContent = `${meta.queryCache.size} entries · ${hitRate}% hit`;

    const fpsElement = byId("m-fps");
    fpsElement.textContent = String(meta.fps);
    fpsElement.className = `value ${meta.fps > 45 ? "good" : meta.fps > 30 ? "warn" : "bad"}`;

    byId("m-lod").textContent = meta.quality.toUpperCase();
    byId("m-hot-count").textContent = String(meta.hotList.length);

    const hotList = byId("m-hot-list");
    hotList.replaceChildren();
    if (!meta.hotList.length) {
        hotList.append("🔍 ");
        const none = document.createElement("span");
        none.textContent = "none yet";
        hotList.append(none);
    } else {
        meta.hotList.forEach((name, index) => {
            if (index) hotList.append(" · ");
            const item = document.createElement("span");
            item.textContent = name;
            hotList.append(item);
        });
    }
}

byId("chat-send").addEventListener("click", handleChat);
byId("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") handleChat();
});
byId("chat-header").addEventListener("click", toggleChat);
byId("chat-header").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleChat();
    }
});
byId("meta-toggle").addEventListener("click", toggleMeta);
byId("refine-now").addEventListener("click", () => {
    refineROMLayout();
    addMessage("🧬 ROM re-encoded in query-heat order. Cache invalidated.", "bot", `refinement ${meta.refinementCounter}`);
});

renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    addMessage("Rendering paused because the WebGL context was lost.", "bot", "awaiting recovery");
});
renderer.domElement.addEventListener("webglcontextrestored", () => {
    addMessage("WebGL context restored.", "bot", "renderer recovered");
});

function animate(time) {
    requestAnimationFrame(animate);
    updateFPS(time);
    updateSun();
    clouds.rotation.y += 0.00015;

    if (flyTarget) {
        const transition = flyTarget;
        transition.progress = Math.min(1, transition.progress + 0.025);
        const progress = transition.progress;
        const eased = progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
        camera.position.lerpVectors(transition.start, transition.end, eased);
        controls.target.set(0, 0, 0);
        if (progress >= 1) flyTarget = null;
    }

    controls.update();
    if (!document.hidden) {
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
    }
}

window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labelRenderer.setSize(width, height);
});

updateMetaPanel();
animate(performance.now());
window.setTimeout(() => {
    addMessage("💡 Try: Where is London? · Time in Delhi · Show cities < 5M", "bot", "query grammar ready");
    refineROMLayout();
}, 500);

window.__earthTwin = Object.freeze({
    query: queryROM,
    snapshot: () => Object.freeze({
        romBytes: meta.romBuffer.byteLength,
        totalQueries: meta.totalQueries,
        cacheHits: meta.cacheHits,
        cacheEntries: meta.queryCache.size,
        fps: meta.fps,
        quality: meta.quality,
        hotList: [...meta.hotList],
        refinements: meta.refinementCounter,
    }),
});

console.info("🧬 EarthTwin Meta-Runtime Online", {
    version: baseROM.version,
    flags: baseROM.flags,
    cities: baseROM.cities.length,
    romBytes: baseROM.bufferSize,
});
