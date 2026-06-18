import * as THREE from 'three';
import type loadMujoco from '@mujoco/mujoco';

type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

export interface MuJoCoScene {
  model: any;
  data: any;
  root: THREE.Group;
  bodies: Record<number, THREE.Group>;
  headBodyId: number;
  pelvisBodyId: number;
}

function getPosition(
  buffer: Float32Array | Float64Array,
  index: number,
  target: THREE.Vector3,
  swizzle = true,
) {
  if (swizzle) {
    return target.set(
      buffer[index * 3 + 0],
      buffer[index * 3 + 2],
      -buffer[index * 3 + 1],
    );
  }
  return target.set(
    buffer[index * 3 + 0],
    buffer[index * 3 + 1],
    buffer[index * 3 + 2],
  );
}

function getQuaternion(
  buffer: Float32Array | Float64Array,
  index: number,
  target: THREE.Quaternion,
  swizzle = true,
) {
  if (swizzle) {
    return target.set(
      -buffer[index * 4 + 1],
      -buffer[index * 4 + 3],
      buffer[index * 4 + 2],
      -buffer[index * 4 + 0],
    );
  }
  return target.set(
    buffer[index * 4 + 0],
    buffer[index * 4 + 1],
    buffer[index * 4 + 2],
    buffer[index * 4 + 3],
  );
}

function getName(namesArray: Uint8Array, adr: number): string {
  const decoder = new TextDecoder('utf-8');
  let end = adr;
  while (end < namesArray.length && namesArray[end] !== 0) end++;
  return decoder.decode(namesArray.subarray(adr, end));
}

export async function loadMuJoCoScene(
  mujoco: MujocoModule,
  xmlPath: string,
): Promise<MuJoCoScene> {
  // Load the model from the Emscripten virtual filesystem path.
  const model = mujoco.MjModel.from_xml_path(xmlPath);
  const data = new mujoco.MjData(model);

  mujoco.mj_forward(model, data);

  const namesArray = new Uint8Array(model.names);
  const bodies: Record<number, THREE.Group> = {};
  const meshes: Record<number, THREE.BufferGeometry> = {};

  const root = new THREE.Group();
  root.name = 'MuJoCo Root';

  const material = new THREE.MeshPhysicalMaterial({ color: 0xffffff });

  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;

    const bodyId = model.geom_bodyid[g];
    const type = model.geom_type[g];
    const geomName = getName(namesArray, model.name_geomadr[g]);
    const size = [
      model.geom_size[g * 3 + 0],
      model.geom_size[g * 3 + 1],
      model.geom_size[g * 3 + 2],
    ];

    if (!bodies[bodyId]) {
      const group = new THREE.Group();
      group.name = getName(namesArray, model.name_bodyadr[bodyId]);
      group.userData.bodyId = bodyId;
      bodies[bodyId] = group;
    }

    let geometry: THREE.BufferGeometry;
    switch (type) {
      case mujoco.mjtGeom.mjGEOM_SPHERE.value:
        geometry = new THREE.SphereGeometry(size[0]);
        break;
      case mujoco.mjtGeom.mjGEOM_CAPSULE.value:
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 8, 16);
        break;
      case mujoco.mjtGeom.mjGEOM_CYLINDER.value:
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0, 16);
        break;
      case mujoco.mjtGeom.mjGEOM_BOX.value:
        geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
        break;
      case mujoco.mjtGeom.mjGEOM_MESH.value: {
        const meshId = model.geom_dataid[g];
        if (!(meshId in meshes)) {
          geometry = buildMeshGeometry(model, meshId);
          meshes[meshId] = geometry;
        } else {
          geometry = meshes[meshId];
        }
        break;
      }
      case mujoco.mjtGeom.mjGEOM_PLANE.value:
        geometry = new THREE.PlaneGeometry(1, 1);
        break;
      default:
        geometry = new THREE.SphereGeometry(size[0] * 0.5);
    }

    const rgba = [
      model.geom_rgba[g * 4 + 0],
      model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2],
      model.geom_rgba[g * 4 + 3],
    ];

    const currentMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      transparent: rgba[3] < 1.0,
      opacity: rgba[3],
      roughness: 0.65,
      metalness: 0.1,
    });

    let mesh: THREE.Mesh | THREE.Group;
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(80, 80),
        new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.9 }),
      );
      plane.rotation.x = -Math.PI / 2;
      plane.receiveShadow = true;
      mesh = plane;
    } else {
      mesh = new THREE.Mesh(geometry, currentMaterial);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }

    mesh.userData.geomId = g;
    mesh.userData.bodyId = bodyId;
    bodies[bodyId].add(mesh);

    getPosition(model.geom_pos, g, mesh.position);
    if (type !== mujoco.mjtGeom.mjGEOM_PLANE.value) {
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }
    if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
  }

  // Add bodies to the root using world transforms from MuJoCo data.
  for (let b = 0; b < model.nbody; b++) {
    if (!bodies[b]) {
      bodies[b] = new THREE.Group();
      bodies[b].name = getName(namesArray, model.name_bodyadr[b]);
      bodies[b].userData.bodyId = b;
    }
    if (b === 0) {
      root.add(bodies[b]);
    } else {
      bodies[0].add(bodies[b]);
    }
  }

  // Ground grid for visual reference.
  const grid = new THREE.GridHelper(40, 80, 0x475569, 0x1e293b);
  grid.position.y = 0.001;
  root.add(grid);

  updateSceneTransforms(model, data, bodies);

  const pelvisBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis');
  let headBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'head_link');
  if (headBodyId < 0) headBodyId = pelvisBodyId;

  return { model, data, root, bodies, headBodyId, pelvisBodyId };
}

function buildMeshGeometry(model: any, meshId: number): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  const vertStart = model.mesh_vertadr[meshId] * 3;
  const vertEnd = (model.mesh_vertadr[meshId] + model.mesh_vertnum[meshId]) * 3;
  const vertexBuffer = new Float32Array(model.mesh_vert.subarray(vertStart, vertEnd));
  for (let v = 0; v < vertexBuffer.length; v += 3) {
    const temp = vertexBuffer[v + 1];
    vertexBuffer[v + 1] = vertexBuffer[v + 2];
    vertexBuffer[v + 2] = -temp;
  }

  const normalAdr = model.mesh_normaladr[meshId];
  const normalNum = model.mesh_normalnum[meshId];
  const hasNormal = normalAdr >= 0 && normalNum > 0;
  const normalBuffer = hasNormal
    ? new Float32Array(model.mesh_normal.subarray(normalAdr * 3, (normalAdr + normalNum) * 3))
    : new Float32Array(0);
  if (hasNormal) {
    for (let v = 0; v < normalBuffer.length; v += 3) {
      const temp = normalBuffer[v + 1];
      normalBuffer[v + 1] = normalBuffer[v + 2];
      normalBuffer[v + 2] = -temp;
    }
  }

  const texCoordAdr = model.mesh_texcoordadr[meshId];
  const texCoordNum = model.mesh_texcoordnum[meshId];
  const hasUV = texCoordAdr >= 0 && texCoordNum > 0;
  const uvBuffer = hasUV
    ? new Float32Array(
        model.mesh_texcoord.subarray(texCoordAdr * 2, (texCoordAdr + texCoordNum) * 2),
      )
    : new Float32Array(0);

  const faceStart = model.mesh_faceadr[meshId] * 3;
  const faceEnd = (model.mesh_faceadr[meshId] + model.mesh_facenum[meshId]) * 3;
  const faceToVertex = new Uint32Array(model.mesh_face.subarray(faceStart, faceEnd));
  const faceToUV = hasUV
    ? new Uint32Array(model.mesh_facetexcoord.subarray(faceStart, faceEnd))
    : new Uint32Array(0);
  const faceToNormal =
    normalNum > 0
      ? new Uint32Array(model.mesh_facenormal.subarray(faceStart, faceEnd))
      : new Uint32Array(0);

  const swizzledUV = new Float32Array((vertexBuffer.length / 3) * 2);
  const swizzledNormal = new Float32Array(vertexBuffer.length);

  for (let t = 0; t < faceToVertex.length / 3; t++) {
    for (let k = 0; k < 3; k++) {
      const vi = faceToVertex[t * 3 + k];
      if (hasUV) {
        const uvi = faceToUV[t * 3 + k];
        swizzledUV[vi * 2 + 0] = uvBuffer[uvi * 2 + 0];
        swizzledUV[vi * 2 + 1] = uvBuffer[uvi * 2 + 1];
      }
      if (hasNormal) {
        const nvi = faceToNormal[t * 3 + k];
        swizzledNormal[vi * 3 + 0] = normalBuffer[nvi * 3 + 0];
        swizzledNormal[vi * 3 + 1] = normalBuffer[nvi * 3 + 1];
        swizzledNormal[vi * 3 + 2] = normalBuffer[nvi * 3 + 2];
      }
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(vertexBuffer, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(swizzledNormal, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(swizzledUV, 2));
  geometry.setIndex(Array.from(faceToVertex));
  geometry.computeVertexNormals();

  return geometry;
}

const tmpVec = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();

export function updateSceneTransforms(
  model: any,
  data: any,
  bodies: Record<number, THREE.Group>,
): void {
  for (let b = 1; b < model.nbody; b++) {
    const body = bodies[b];
    if (!body) continue;
    getPosition(data.xpos, b, body.position);
    getQuaternion(data.xquat, b, body.quaternion);
  }
}

export function getBodyWorldTransform(
  model: any,
  data: any,
  bodyId: number,
): { position: THREE.Vector3; quaternion: THREE.Quaternion } {
  return {
    position: getPosition(data.xpos, bodyId, tmpVec.clone(), true),
    quaternion: getQuaternion(data.xquat, bodyId, tmpQuat.clone(), true),
  };
}
