var rayParams_lightning = {
    sourceOffset: new THREE.Vector3(),
    destOffset: new THREE.Vector3(),
    radius0: 0.05,
    radius1: 0.02,
    minRadius: 2.5,
    maxIterations: 7,
    isEternal: true,

    timeScale: 2,

    propagationTimeFactor: 0.05,
    vanishingTimeFactor: 0.95,
    subrayPeriod: 3.5,
    subrayDutyCycle: 0.6,
    maxSubrayRecursion: 3,
    ramification: 5,
    recursionProbability: 0.6,

    roughness: 0.93,
    straightness: 0.8
};
var rayParams_beam = {
    sourceOffset: new THREE.Vector3(),
    destOffset: new THREE.Vector3(),
    radius0: 0.05,
    radius1: 0.01,
    minRadius: 2.5,
    maxIterations: 7,
    isEternal: true,

    timeScale: 2,

    propagationTimeFactor: 0.05,
    vanishingTimeFactor: 0.95,
    subrayPeriod: 3.5,
    subrayDutyCycle: 0.6,
    maxSubrayRecursion: 3,
    ramification: 0,
    recursionProbability: 0,

    roughness: 0,
    straightness: 1.0
};
Array.prototype.indexOf = function(val) {
    for (var i = 0; i < this.length; i++) {
        if (this[i] == val) return i;
    }
    return -1;
};
    
Array.prototype.remove = function(val) {
    var index = this.indexOf(val);
    if (index > -1) {
        this.splice(index, 1);
    }
};
function parse_update_env(buf_str) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_env") != -1){
            parse_env(str)
        }
    }
}

// all init cmd must only be executed once
var executed_init_cmd = {}
function parse_init(buf_str) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_style") != -1){
            if (executed_init_cmd[str]) {continue;} else{executed_init_cmd[str]=true;}
            parse_style(str)
        }
        if(str.search(">>geometry_rotate_scale_translate") != -1){
            if (executed_init_cmd[str]) {continue;} else{executed_init_cmd[str]=true;}
            parse_geometry(str)
        }
        if(str.search(">>advanced_geometry_rotate_scale_translate") != -1){
            if (executed_init_cmd[str]) {continue;} else{executed_init_cmd[str]=true;}
            parse_advanced_geometry(str)
        }
        if(str.search(">>advanced_geometry_material") != -1){
            if (executed_init_cmd[str]) {continue;} else{executed_init_cmd[str]=true;}
            parse_advanced_geometry_material(str)
        }
        
    }
}

function clear_everything(){
    // remove terrain surface
    // if (init_terrain){
    //     init_terrain = false;
    //     detach_dispose(window.glb.terrain_mesh, from_where=window.glb.scene);
    //     window.glb.scene.remove(window.glb.terrain_mesh);
    // }

    // remove all objects
    for (let i = window.glb.core_Obj.length-1; i>=0 ; i--) {
        purge_core_obj(i) // will be removed from window.glb.core_Obj
    }

    // clear flash
    for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
        detach_dispose(window.glb.flash_Obj[i]['mesh'], from_where=window.glb.scene);
        window.glb.flash_Obj[i]['mesh'] = null;
        window.glb.flash_Obj[i]['valid'] = false;
    }
    // line 3d
    for (let i = window.glb.line_Obj.length-1; i>=0; i--) {
        detach_dispose(window.glb.line_Obj[i]['mesh'], from_where=window.glb.scene);
        window.glb.line_Obj[i]['mesh'] = null;
        window.glb.line_Obj[i]['valid'] = false;
    }

}




var init_terrain = false;
var TerrainMaterialKargs = {}

function parse_env(str){
    let re_style = />>set_env\('(.*?)'/
    let re_res = str.match(re_style)
    let style = re_res[1]

    if(style=="terrain"){
        let get_theta = />>set_env\('terrain',theta=([^,)]*)/
        let get_theta_res = str.match(get_theta)
        let terrain_A   = match_karg(str, 'terrain_A',   'float', null)  
        let terrain_B   = match_karg(str, 'terrain_B',   'float', null)  
        let show_lambda = match_karg(str, 'show_lambda', 'float', null)        
        let theta = parseFloat(get_theta_res[1])
        
        // 投射阴影
        for (let i = window.glb.core_Obj.length-1; i>=0 ; i--) {
            window.glb.core_Obj[i].castShadow = true;
        }
    
    
        ////////////////////// add terrain /////////////////////
        let width = 13; let height = 13;
        let Segments = 200; let need_remove_old = false;
        if (!init_terrain){
            init_terrain=true; need_remove_old = false;
            if(!TerrainMaterialKargs['map']){
                let texture = THREE.ImageUtils.loadTexture('/wget/dirt2.jpg');
                texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
                texture.repeat.set(8, 8);
                // TerrainMaterialKargs['map'] = texture;
                TerrainMaterialKargs['color'] = 'Sienna';
                TerrainMaterialKargs['bumpMap'] = texture;
                TerrainMaterialKargs['bumpScale'] = 0.01;
            }
            window.glb.renderer.shadowMap.enabled = true;
            var light = new THREE.DirectionalLight(0xffffff,0.9);
            light.castShadow = true;
            light.position.set(0,50,7);
            window.glb.scene.add(light);
            window.glb.camera.children[0].visible = false;
            window.glb.terrain_material = new THREE.MeshPhongMaterial(TerrainMaterialKargs);

        }else{
            need_remove_old = true;

        }
        let geometry = new THREE.PlaneBufferGeometry(width, height, Segments - 1, Segments - 1); //(width, height,widthSegments,heightSegments)
        geometry.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
        let array = geometry.attributes.position.array;
        for (let i = 0; i < Segments * Segments; i++) {
            let x = array[i * 3 + 0];
            let _x_ = array[i * 3 + 0];
            let z = array[i * 3 + 2];
            let _y_ = -array[i * 3 + 2];
    
            let A=0.05; if(terrain_A){A=terrain_A};
            let B=0.2; if(terrain_B){B=terrain_B};
            let lambda=4; if(show_lambda){lambda=show_lambda};
            let X_ = _x_*Math.cos(theta) + _y_*Math.sin(theta);
            let Y_ = -_x_*Math.sin(theta) + _y_*Math.cos(theta);
            let Z = -1 +B*( (0.1*X_) ** 2 + (0.1*Y_) ** 2 )- A * Math.cos(2 * Math.PI * (0.3*X_))  - A * Math.cos(2 * Math.PI * (0.5*Y_))
            Z = -Z;
            Z = (Z-1)*lambda;
            Z = Z - 0.1
            array[i * 3 + 1] = Z
        }
        geometry.computeBoundingSphere(); 
        geometry.computeVertexNormals();
        if (need_remove_old){
            window.glb.scene.remove(window.glb.terrain_mesh);
            window.glb.terrain_mesh.geometry.dispose();
        }
        window.glb.terrain_mesh = new THREE.Mesh(geometry, window.glb.terrain_material);
        window.glb.terrain_mesh.receiveShadow = true;
        window.glb.scene.add(window.glb.terrain_mesh);

        console.log('update terrain')
    }
    if(style=="terrain_rm"){
        if (!init_terrain){
        }else{
            detach_dispose(window.glb.terrain_mesh, from_where=window.glb.scene);
            window.glb.terrain_mesh = null;
        }
        
    }
    if(style=="clear_everything"){
        clear_everything();
    }
    if(style=="clear_track"){
        for (let i = window.glb.core_Obj.length-1; i>=0 ; i--) {
            let object = window.glb.core_Obj[i]
            // 初始化历史轨迹
            object.his_positions = [];
            for ( let i = 0; i < MAX_HIS_LEN; i ++ ) {
                object.his_positions.push( new THREE.Vector3(object.next_pos.x, object.next_pos.y, object.next_pos.z) );
            }
        }
    }
    if(style=="clear_flash"){
        for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
            detach_dispose(window.glb.flash_Obj[i]['mesh'], from_where=window.glb.scene);
            window.glb.flash_Obj[i]['mesh'] = null;
            window.glb.flash_Obj[i]['valid'] = false;
        }
    }
}


function parse_style(str){
    //E.g. >>flash('lightning',src=0.00000000e+00,dst=1.00000000e+01,dur=1.00000000e+00)
    let re_style = />>set_style\('(.*?)'/
    let re_res = str.match(re_style)
    let style = re_res[1]
    if(style=="terrain"){
        console.log('use set_env')
    }else if (style=="vhmap_buffer_size"){
        let vhmap_buffer_size = match_karg(str, 'size', 'int', null)
        if(vhmap_buffer_size){

            window.glb.buffer_size = vhmap_buffer_size
        }
    }
    else if (style=="grid3d"){
        let gridXZ = new THREE.GridHelper(1000, 10, 0xEED5B7, 0xEED5B7);
        gridXZ.position.set(500,0,500);
        window.glb.scene.add(gridXZ);
        let gridXY = new THREE.GridHelper(1000, 10, 0xEED5B7, 0xEED5B7);
        gridXY.position.set(500,500,0);
        gridXY.rotation.x = Math.PI/2;
        window.glb.scene.add(gridXY);
        let gridYZ = new THREE.GridHelper(1000, 10, 0xEED5B7, 0xEED5B7);
        gridYZ.position.set(0,500,500);
        gridYZ.rotation.z = Math.PI/2;
        window.glb.scene.add(gridYZ);
    }
    else if (style=="grid"){
        window.glb.scene.children.filter(function (x){return (x.type == 'GridHelper')}).forEach(function(x){
            x.visible = true
        })
    }
    else if(style=="nogrid"){
        window.glb.scene.children.filter(function (x){return (x.type == 'GridHelper')}).forEach(function(x){
            x.visible = false
        })
    }else if(style=="gray"){
        window.glb.scene.background = new THREE.Color(0xa0a0a0);
    }else if(style=="background"){
        let bgcolor = match_karg(str, 'color', 'str', null)
        if (bgcolor){window.glb.scene.background = new THREE.Color(bgcolor);}
    }else if(style=='star'){
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        for ( let i = 0; i < 10000; i ++ ) {
            let x;
            let y;
            let z;
            while (true){
                x = THREE.MathUtils.randFloatSpread( 2000 );
                y = THREE.MathUtils.randFloatSpread( 2000 );
                z = THREE.MathUtils.randFloatSpread( 2000 );
                if ((x*x+y*y+z*z)>20000){break;} // not too close to the center
            }
            vertices.push( x ); // x
            vertices.push( y ); // y
            vertices.push( z ); // z
        }
        geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );
        const particles = new THREE.Points( geometry, new THREE.PointsMaterial( { color: 0x888888 } ) );
        window.glb.scene.add( particles );
    }else if(style=='many star'){
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        for ( let i = 0; i < 50000; i ++ ) {
            let x;
            let y;
            let z;
            while (true){
                x = THREE.MathUtils.randFloatSpread( 20000 );
                y = THREE.MathUtils.randFloatSpread( 20000 );
                z = THREE.MathUtils.randFloatSpread( 20000 );
                if ((x*x+y*y+z*z)>20000){break;} // not too close to the center
            }
            vertices.push( x ); // x
            vertices.push( y ); // y
            vertices.push( z ); // z
        }
        geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );
        const particles = new THREE.Points( geometry, new THREE.PointsMaterial( { color: 0x888888 } ) );
        window.glb.scene.add( particles );
    }else if(style=='earth'){
        var onRenderFcts= [];
        var light	= new THREE.AmbientLight( 0x222222 )
        window.glb.scene.add( light )
    
        var light	= new THREE.DirectionalLight( 0xffffff, 1 )
        light.position.set(5,5,5)
        window.glb.scene.add( light )
        light.castShadow	= true
        light.shadowCameraNear	= 0.01
        light.shadowCameraFar	= 15
        light.shadowCameraFov	= 45
    
        light.shadowCameraLeft	= -1
        light.shadowCameraRight	=  1
        light.shadowCameraTop	=  1
        light.shadowCameraBottom= -1
        // light.shadowCameraVisible	= true
    
        light.shadowBias	= 0.001
        light.shadowDarkness	= 0.2
    
        light.shadowMapWidth	= 1024
        light.shadowMapHeight	= 1024

        var containerEarth	= new THREE.Object3D()
        containerEarth.rotateZ(-23.4 * Math.PI/180)
        containerEarth.position.z	= -50
        containerEarth.scale.x	= 50
        containerEarth.scale.y	= 50
        containerEarth.scale.z	= 50
        window.glb.scene.add(containerEarth)
    
        var earthMesh	= THREEx.Planets.createEarth()
        earthMesh.receiveShadow	= true
        earthMesh.castShadow	= true
        containerEarth.add(earthMesh)
        onRenderFcts.push(function(delta, now){
            earthMesh.rotation.y += 1/32 * delta;		
        })
    
        var geometry	= new THREE.SphereGeometry(0.5, 32, 32)
        var material	= THREEx.createAtmosphereMaterial()
        material.uniforms.glowColor.value.set(0x00b3ff)
        material.uniforms.coeficient.value	= 0.8
        material.uniforms.power.value		= 2.0
        var mesh	= new THREE.Mesh(geometry, material );
        mesh.scale.multiplyScalar(1.01);
        containerEarth.add( mesh );
        // new THREEx.addAtmosphereMaterial2DatGui(material, datGUI)
    
        var geometry	= new THREE.SphereGeometry(0.5, 32, 32)
        var material	= THREEx.createAtmosphereMaterial()
        material.side	= THREE.BackSide
        material.uniforms.glowColor.value.set(0x00b3ff)
        material.uniforms.coeficient.value	= 0.5
        material.uniforms.power.value		= 4.0
        var mesh	= new THREE.Mesh(geometry, material );
        mesh.scale.multiplyScalar(1.15);
        containerEarth.add( mesh );
        window.glb.renderer.shadowMapEnabled = true
    }else if(style=="font"){
        let fontPath = match_karg(str, 'fontPath', 'str', null)
        let fontLineHeight = match_karg(str, 'fontLineHeight', 'int', null)
        if (fontPath){
            const loader = new window.glb.import_TTFLoader();
            loader.load( fontPath, function ( json ) {
                window.glb.font = new window.glb.import_Font( json );
                if (fontLineHeight){window.glb.font.data.defLineHeight = fontLineHeight}
            } );
        }
    }else if(style=="sky"){
        async function loadsky() {
            let SkyMod = await import('./jsm/objects/Sky.js');
            const effectController = {
                turbidity: 10,
                rayleigh: 3,
                mieCoefficient: 0.005,
                mieDirectionalG: 0.7,
                elevation: 2,
                azimuth: 180,
                exposure: window.glb.renderer.toneMappingExposure
            };
            sky = new SkyMod.Sky();
            sky.scale.setScalar( 450000 );
            window.glb.scene.add( sky );
            sun = new THREE.Vector3();
            const uniforms = sky.material.uniforms;
            uniforms[ 'turbidity' ].value = effectController.turbidity;
            uniforms[ 'rayleigh' ].value = effectController.rayleigh;
            uniforms[ 'mieCoefficient' ].value = effectController.mieCoefficient;
            uniforms[ 'mieDirectionalG' ].value = effectController.mieDirectionalG;
            const phi = THREE.MathUtils.degToRad( 90 - effectController.elevation );
            const theta = THREE.MathUtils.degToRad( effectController.azimuth );
            sun.setFromSphericalCoords( 1, phi, theta );
            uniforms[ 'sunPosition' ].value.copy( sun );
            window.glb.renderer.toneMappingExposure = effectController.exposure;
        }
        loadsky()

    }else if(style=="skybox"){
        async function loadskybox() {
            const textureLoader = new THREE.TextureLoader();
            let skyboxPath = match_karg(str, 'path', 'str', null)
            if(skyboxPath){
                textureEquirec = textureLoader.load( skyboxPath );
                textureEquirec.mapping = THREE.EquirectangularReflectionMapping;
                // textureEquirec.encoding = THREE.sRGBEncoding;
                window.glb.scene.background = textureEquirec;
                // const ambient = new THREE.AmbientLight( 0xffffff );
				// window.glb.scene.add( ambient );
                // window.glb.renderer.outputEncoding = THREE.sRGBEncoding;
            }else{
                alert('Skybox path not given! Please use path=xxxx !')
            }
        }
        loadskybox()

    }else if(style=="skybox6side"){
        async function loadskybox6side() {
            const loader = new THREE.CubeTextureLoader();
            // const textureLoader = new THREE.TextureLoader();
            let posx = match_karg(str, 'posx', 'str', null)
            let negx = match_karg(str, 'negx', 'str', null)
            let posy = match_karg(str, 'posy', 'str', null)
            let negy = match_karg(str, 'negy', 'str', null)
            let posz = match_karg(str, 'posz', 'str', null)
            let negz = match_karg(str, 'negz', 'str', null)
            if(posx){
                let textureCube = loader.load( [ posx, negx, posy, negy, posz, negz ] );
                // textureCube.encoding = THREE.sRGBEncoding;
                window.glb.scene.background = textureCube;
                // const ambient = new THREE.AmbientLight( 0x020202 );
				// window.glb.scene.add( ambient );
                // window.glb.renderer.outputEncoding = THREE.sRGBEncoding;
            }else{
                alert('Skybox path not given! Please use path=xxxx !')
            }
        }
        loadskybox6side()

    }
    else{
        alert('style not understood:'+str)
    }
}

function geo_transform(geometry, ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z){
    geometry.rotateX(ro_x);
    geometry.rotateY(ro_y);
    geometry.rotateZ(ro_z);
    geometry.scale(scale_x,scale_y,scale_z)
    geometry.translate(trans_x, trans_y, trans_z)
    return geometry
}

function parse_advanced_geometry(str){
    const pattern = get_reg_exp(">>advanced_geometry_rotate_scale_translate\\('(.*?)','(.*?)',([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)(.*)\\)")
    let match_res = str.match(pattern)
    let name = match_res[1]
    let build_cmd = match_res[2]
    let ro_x = parseFloat(match_res[3])
    // z --> y, y --- z reverse z axis and y axis
    let ro_y = parseFloat(match_res[5])
    // z --> y, y --- z reverse z axis and y axis
    let ro_z = -parseFloat(match_res[4])

    let scale_x = parseFloat(match_res[6])
    // z --> y, y --- z reverse z axis and y axis
    let scale_y = parseFloat(match_res[8])
    // z --> y, y --- z reverse z axis and y axis
    let scale_z = parseFloat(match_res[7])

    let trans_x = parseFloat(match_res[9])
    // z --> y, y --- z reverse z axis and y axis
    let trans_y = parseFloat(match_res[11])
    // z --> y, y --- z reverse z axis and y axis
    let trans_z = -parseFloat(match_res[10])

    // load geo
    window.glb.base_geometry[name] = null;
    if(build_cmd.includes('fbx=')){
        let each_part = build_cmd.split('=')
        let path_of_fbx_file = each_part[1]
        const loader = new window.glb.import_FBXLoader();
        window.glb.base_geometry[name] = 'loading'
        loader.load(path_of_fbx_file, function ( object ) {
            window.glb.base_geometry[name] = object.children[0].geometry;
            window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
        });
    }else if(build_cmd.includes('gltf=')){
        // let each_part = build_cmd.split('=')
        // let path_of_gltf_file = each_part[1]

        // async function loadGLTFLoader(src) {
        //     window.glb.base_geometry[name] = 'loading'
        //     let Mod = await import(src);
        //     const loader = new Mod.GLTFLoader();
        //     loader.load( path_of_gltf_file, function ( gltf ) {
        //         let gltf_group = gltf.scene; //
        //         let obj_list = []
        //         gltf.scene.traverse(function (child) {
        //             if (child.geometry){
        //                 child.updateMatrix();
        //                 obj_list.push(child)
        //             }
        //         });

        //         let geo = mergeBufferGeometry(obj_list)

        //         window.glb.base_geometry[name] = geo;
        //         window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
        //     }, undefined, function ( e ) {
        //         console.error( e );
        //     } );
        // }
        // loadGLTFLoader(`/examples/jsm/loaders/GLTFLoader.js`)
    }else{
        // very basic shapes
        window.glb.base_geometry[name] = eval('new THREE.'+build_cmd)
        window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
    }
}

function parse_advanced_geometry_material(str){
    const pattern = get_reg_exp(">>advanced_geometry_material\\('(.*?)'")
    // const pattern = />>advanced_geometry_material\('(.*?)'/
    let match_res = str.match(pattern)
    let name = match_res[1]
    if (!window.glb.base_geometry[name]){
        alert('[advanced_geometry_material]: Missing the geometry of '+name)
    }
    window.glb.base_material[name] = null;
    kargs = {}
    let map = match_karg(str, 'map', 'str', null)
    let bumpMap = match_karg(str, 'bumpMap', 'str', null)
    let bumpScale = match_karg(str, 'bumpScale', 'float', null)
    let specularMap = match_karg(str, 'specularMap', 'str', null)
    let specular = match_karg(str, 'specular', 'str', null)

    if (map){kargs['map'] = THREE.ImageUtils.loadTexture(map)}
    if (bumpMap){kargs['bumpMap'] = THREE.ImageUtils.loadTexture(bumpMap)}
    if (bumpScale){kargs['bumpScale'] = bumpScale}
    if (specularMap){kargs['specularMap'] = THREE.ImageUtils.loadTexture(specularMap)}
    if (specular){kargs['specular'] = new THREE.Color(specular)}

    window.glb.base_material[name] = new THREE.MeshPhongMaterial(kargs)

}

function parse_geometry(str){
    const pattern = get_reg_exp(">>geometry_rotate_scale_translate\\('(.*?)',([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)(.*)\\)")
    let match_res = str.match(pattern)
    let name = match_res[1]
    let ro_x = parseFloat(match_res[2])
    // z --> y, y --- z reverse z axis and y axis
    let ro_y = parseFloat(match_res[4])
    // z --> y, y --- z reverse z axis and y axis
    let ro_z = -parseFloat(match_res[3])

    let scale_x = parseFloat(match_res[5])
    // z --> y, y --- z reverse z axis and y axis
    let scale_y = parseFloat(match_res[7])
    // z --> y, y --- z reverse z axis and y axis
    let scale_z = parseFloat(match_res[6])

    let trans_x = parseFloat(match_res[8])
    // z --> y, y --- z reverse z axis and y axis
    let trans_y = parseFloat(match_res[10])
    // z --> y, y --- z reverse z axis and y axis
    let trans_z = -parseFloat(match_res[9])

    let lib = {
        'monkey':'examples/models/json/suzanne_buffergeometry.json'
    }
    let path = lib[name]

    // load geo
    if (window.glb.base_geometry[name]==null){
        window.glb.base_geometry[name] = null;
        // very basic shapes
        if (name=='box'){
            window.glb.base_geometry[name] = new THREE.BoxGeometry(1, 1, 1);
            window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
        }else if(name=='sphe' || name=='ball'){
            window.glb.base_geometry[name] = new THREE.SphereGeometry(1);
            window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
        }else if(name=='cone'){
            window.glb.base_geometry[name] = new THREE.ConeGeometry(1, 2*1);
            window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
        }else{
            // other shapes in lib
            const loader = new THREE.BufferGeometryLoader();
            window.glb.base_geometry[name] = 'loading'
            loader.load(path, function (geometry) {
                geometry.computeVertexNormals();
                window.glb.base_geometry[name] = geo_transform(geometry, ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
            });
        }
    }else{
        window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
    }

}
function parse_update_core(buf_str, pp) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if (str.search(">>v2dx") != -1) {
            // name, xpos, ypos, zpos, dir=0, **kargs
            parse_core_obj(str)
        }
    }
}

function parse_update_flash(buf_str) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if(str.search(">>flash") != -1){
            parse_flash(str)
        }
        if(str.search(">>line3d") != -1){
            parse_line(str)
        }
    }
}

var reg_exp = {}
function get_reg_exp(exp){
    if(!reg_exp[exp]){
        reg_exp[exp] = new RegExp(exp);
    }
    return reg_exp[exp];
}
var reg_exp_g = {}
function get_reg_exp_g(exp){
    if(!reg_exp_g[exp]){
        reg_exp_g[exp] = new RegExp(exp, 'g');
    }
    return reg_exp_g[exp];
}
function match_karg(str, key, type, defaultValue){
    let res = null;
    if (type=='float'){
        let reg_exp = get_reg_exp(key + "=([^,)]*)");
        // let reg_exp = new RegExp(key + "=([^,)]*)")
        let _RE = str.match(reg_exp);
        res = (!(_RE === null))?parseFloat(_RE[1]):defaultValue;
    }
    if (type=='int'){
        let reg_exp = get_reg_exp(key + "=([^,)]*)");
        // let reg_exp = new RegExp(key + "=([^,)]*)");
        let _RE = str.match(reg_exp);
        res = (!(_RE === null))?parseInt(_RE[1]):defaultValue;
    }
    if (type=='str'){
        let reg_exp = get_reg_exp(key + "='(.*?)'");
        // let reg_exp = new RegExp(key + "='(.*?)'");
        let _RE = str.match(reg_exp);
        res = (!(_RE === null))?_RE[1]:defaultValue;
        if (!(_RE === null)){
            res = res.replace(/\$/g,"\n");
            // res = res.replace("$", "\n"); 不能用replace，replace只能替换一次
        }
    }
    if (type=='arr_float'){
        let reg_exp = get_reg_exp(key + "=\\[(.*?)\\]");
        // let reg_exp = new RegExp(key + "=\\[(.*?)\\]");
        let _RE = str.match(reg_exp);
        if (_RE === null){
            res = defaultValue;
        }else{
            res = _RE[1].split(',');
            for (i=0; i<res.length; i++){
                res[i] = parseFloat(res[i]);
            }
        }
    }
    return res
}

function parse_line(str){
    const pattern = />>line3d\('(.*?)',(.*)\)/
    let match_res = str.match(pattern)
    let name = match_res[1] // norm|0|Violet|0.010

    let name_split = name.split('|')
    let type = name_split[0] // norm
    let my_id = name_split[1] // 0
    let color_str = name_split[2]   //Violet
    let size = parseFloat(name_split[3]) //0.010


    let x_arr = match_karg(str, 'x_arr', 'arr_float', null)
    let y_arr = match_karg(str, 'z_arr', 'arr_float', null)
    let z_arr = match_karg(str, 'y_arr', 'arr_float', null) 
    if(!x_arr || !y_arr || !z_arr){alert('Cannot parse line3d, x/y/z_arr missing:', str)}
    for (i=0; i<z_arr.length; i++){z_arr[i] = -z_arr[i]}


    // find core obj by my_id
    let object = find_lineobj_by_id(my_id)
    let parsed_obj_info = {} 
    parsed_obj_info['name'] = name
    parsed_obj_info['x_arr'] = x_arr  
    parsed_obj_info['y_arr'] = y_arr
    parsed_obj_info['z_arr'] = z_arr

    parsed_obj_info['type'] = type  
    parsed_obj_info['tension'] = match_karg(str, 'tension', 'float', 1)
    parsed_obj_info['my_id'] = my_id  
    parsed_obj_info['color_str'] = color_str  
    parsed_obj_info['size'] = size
    parsed_obj_info['label_marking'] = match_karg(str, 'label', 'str', `id ${my_id}`)
    parsed_obj_info['label_color'] = match_karg(str, 'label_color', 'str', 'black')
    parsed_obj_info['opacity'] = match_karg(str, 'opacity', 'float', 1)


    parsed_obj_info['dashScale'] = match_karg(str, 'dashScale', 'float', null)
    parsed_obj_info['dashSize'] = match_karg(str, 'dashSize', 'float', null)
    parsed_obj_info['gapSize'] = match_karg(str, 'gapSize', 'float', null)

    if(type=='simple'){
        apply_simple_line_update(object, parsed_obj_info)
    }
    if(type=='fat'){
        apply_line_update(object, parsed_obj_info)
    }
}

function find_lineobj_by_id(my_id){
    for (let i = 0; i < window.glb.line_Obj.length; i++) {
        if (window.glb.line_Obj[i].my_id == my_id) {
            return window.glb.line_Obj[i];
        }
    }
    return null
}
// function find_text_by_id(my_id){
//     for (let i = 0; i < window.glb.text_Obj.length; i++) {
//         if (window.glb.text_Obj[i].my_id == my_id) {
//             return window.glb.text_Obj[i];
//         }
//     }
//     return null
// }

var dictionary_id2index = {}
function find_obj_by_id(my_id){
    let string_id = my_id.toString()
    if (dictionary_id2index[string_id]!=null){
        let i = dictionary_id2index[string_id];
        if (window.glb.core_Obj[i]!=null && window.glb.core_Obj[i].my_id == my_id) {
            return window.glb.core_Obj[i];
        }
    }

    // the usual way
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        if (window.glb.core_Obj[i].my_id == my_id) {
            dictionary_id2index[string_id] = i;
            return window.glb.core_Obj[i];
        }
    }
    return null
}

//修改颜色
function changeCoreObjColor(object, color_str){
    const colorjs = color_str;
    object.material.color.set(colorjs)
    object.color_str = color_str;
}

const ARC_SEGMENTS = 5120;
function apply_line_update(object, parsed_obj_info){
    if (object) {
        // update pos
        const positions = [];
        const positions_catmull = [];
        const point_tmp = new THREE.Vector3();
        for ( let i = 0; i < parsed_obj_info['x_arr'].length; i ++ ) {
            positions.push( new THREE.Vector3(parsed_obj_info['x_arr'][i], parsed_obj_info['y_arr'][i], parsed_obj_info['z_arr'][i]) );
        }
        //load pos into curve
        let curve = object
        curve.points = positions
        if(curve.current_color!=parsed_obj_info['color_str']){
            curve.current_color=parsed_obj_info['color_str']
            changeCoreObjColor(curve.mesh_line, parsed_obj_info['color_str'])
        }
        curve.tension = parsed_obj_info['tension'];
        // calculate positions_catmull
        const divisions = Math.round( 12 * positions.length );
        for ( let i = 0; i < divisions; i ++ ) {
            const t = i / ( divisions - 1 );
            curve.getPoint( t, point_tmp ); 
            positions_catmull.push( point_tmp.x, point_tmp.y, point_tmp.z)
        }
        //apply calculated positions_catmull
        curve.mesh_line.geometry.setPositions( positions_catmull );
        curve.mesh_line.computeLineDistances();
        curve.mesh_line.geometry.computeBoundingSphere();
        curve.mesh_line.geometry.attributes.position.needsUpdate=true
        // curve.mesh_line.geometry.position.needsUpdate = true
    }
    else {

        const positions = [];
        const positions_catmull = [];
        const point_tmp = new THREE.Vector3();
        for ( let i = 0; i < parsed_obj_info['x_arr'].length; i ++ ) {
            positions.push( new THREE.Vector3(parsed_obj_info['x_arr'][i], parsed_obj_info['y_arr'][i], parsed_obj_info['z_arr'][i]) );
        }
        // init CatmullRomCurve3
        const curve = new THREE.CatmullRomCurve3( positions );
        // confirm id and curveType
        curve.my_id = parsed_obj_info['my_id'];
        curve.curveType = 'catmullrom';
        curve.current_color = parsed_obj_info['color_str']
        curve.tension = parsed_obj_info['tension'];
        // load positions_catmull
        const divisions = Math.round( 12 * positions.length );
        for ( let i = 0; i < divisions; i ++ ) {
            const t = i / ( divisions - 1 );
            curve.getPoint( t, point_tmp ); 
            positions_catmull.push( point_tmp.x, point_tmp.y, point_tmp.z)
        }
        // geo
        const geometry = new window.glb.import_LineGeometry();
        geometry.setPositions( positions_catmull );
        // material
        let dashed_ = false
        if (parsed_obj_info['dashScale']){
            dashed_ = true
        }
        matLine = new window.glb.import_LineMaterial( {
            color: parsed_obj_info['color_str'],
            linewidth: parsed_obj_info['size'], // in world units with size attenuation, pixels otherwise
            worldUnits: true,
            opacity: parsed_obj_info['opacity'],
            dashed: dashed_,
            dashScale: parsed_obj_info['dashScale'],
            dashSize: parsed_obj_info['dashSize'],
            gapSize: parsed_obj_info['gapSize'],
            transparent: (parsed_obj_info['opacity']!=1)
        } );
        // matLine.dashed = val;
        // matLine.dashScale = val;
        // matLine.dashSize = 2;
        // matLine.gapSize = 1;
        curve.mesh_line = new window.glb.import_Line2( geometry, matLine );
        curve.mesh_line.geometry.computeBoundingSphere();
        curve.mesh_line.computeLineDistances();
        window.glb.scene.add(curve.mesh_line);
        window.glb.line_Obj.push(curve);
    }

}


function apply_simple_line_update(object, parsed_obj_info){
    if (object) {
        let curve = object
        let x_arr = parsed_obj_info['x_arr']  
        let y_arr = parsed_obj_info['y_arr']
        let z_arr = parsed_obj_info['z_arr']
        const positions = [];
        for ( let i = 0; i < x_arr.length; i ++ ) {
            positions.push( new THREE.Vector3(x_arr[i], y_arr[i], z_arr[i]) );
        }
        curve.points = positions
        const position = curve.mesh.geometry.attributes.position;
        const point = new THREE.Vector3();
        for ( let i = 0; i < ARC_SEGMENTS; i ++ ) {
            const t = i / ( ARC_SEGMENTS - 1 );
            curve.getPoint( t, point );
            position.setXYZ( i, point.x, point.y, point.z );
        }
        if(curve.current_color!=parsed_obj_info['color_str']){
            curve.current_color=parsed_obj_info['color_str']
            changeCoreObjColor(curve.mesh, parsed_obj_info['color_str'])
        }
        curve.mesh.geometry.computeBoundingSphere();
        
        position.needsUpdate = true;
    }
    else {
        let x_arr = parsed_obj_info['x_arr']  
        let y_arr = parsed_obj_info['y_arr']
        let z_arr = parsed_obj_info['z_arr']
        const positions = [];
        for ( let i = 0; i < x_arr.length; i ++ ) {
            positions.push( new THREE.Vector3(x_arr[i], y_arr[i], z_arr[i]) );
        }
        const curve = new THREE.CatmullRomCurve3( positions );
        curve.my_id = parsed_obj_info['my_id'];
        curve.curveType = 'centripetal';
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( ARC_SEGMENTS * 3 ), 3 ) );
        curve.mesh = new THREE.Line( geometry.clone(), new THREE.LineBasicMaterial( {
            color: parsed_obj_info['color_str'],
            // opacity: 0.1,
            transparent: false
        } ) );
        curve.current_color = parsed_obj_info['color_str']
        curve.mesh.castShadow = false;
        // curve.tension = 0.1;
        const position = curve.mesh.geometry.attributes.position;
        const point = new THREE.Vector3();
        for ( let i = 0; i < ARC_SEGMENTS; i ++ ) {
            const t = i / ( ARC_SEGMENTS - 1 );
            curve.getPoint( t, point );
            position.setXYZ( i, point.x, point.y, point.z );
        }
        curve.mesh.geometry.computeBoundingSphere();
        window.glb.scene.add(curve.mesh);
        window.glb.line_Obj.push(curve);
    }

}



function parse_flash(str){
    //E.g. >>flash('lightning',src=0.00000000e+00,dst=1.00000000e+01,dur=1.00000000e+00,color='red')
    let re_type = />>flash\('(.*?)'/
    let re_res = str.match(re_type)
    let type = re_res[1]
    // src
    let re_src = /src=([^,)]*)/
    re_res = str.match(re_src)
    let src = parseInt(re_res[1])
    // dst
    let re_dst = /dst=([^,)]*)/
    re_res = str.match(re_dst)
    let dst = parseInt(re_res[1])
    // dur
    let re_dur = /dur=([^,)]*)/
    re_res = str.match(re_dur)
    let dur = parseFloat(re_res[1])
    // size
    let re_size = /size=([^,)]*)/
    re_res = str.match(re_size)
    let size = parseFloat(re_res[1])
    // color
    let re_color = /color='(.*?)'/
    re_res = str.match(re_color)
    let color = re_res[1]
    make_flash(type, src, dst, dur, size, color)
}
function make_flash(type, src, dst, dur, size, color){
    if (type=='lightning'){
        let rayParams_new = Object.create(rayParams_lightning);
        let src_obj = find_obj_by_id(src); if (!src_obj){return};
        let dst_obj = find_obj_by_id(dst); if (!dst_obj){return};
        rayParams_new.sourceOffset =  src_obj.position;
        rayParams_new.destOffset =    dst_obj.position;
        rayParams_new.radius0 = size
        rayParams_new.radius1 = size/4.0
        if (isNaN(src_obj.position.x) || isNaN(src_obj.position.y)){return}
        // let lightningColor = new THREE.Color( 0xFFB0FF );
        let lightningStrike = new window.glb.import_LightningStrike( rayParams_new );
        let lightningMaterial = new THREE.MeshBasicMaterial( { color: color } );
        let lightningStrikeMesh = new THREE.Mesh( lightningStrike, lightningMaterial );
        window.glb.scene.add( lightningStrikeMesh );
        window.glb.flash_Obj.push({
            'create_time':new Date(),
            'dur':dur,
            'valid':true,
            'mesh':lightningStrikeMesh,
            'update_target':lightningStrike,
        })
    }else if (type=='beam'){
        let rayParams_new = Object.create(rayParams_beam);
        let src_obj = find_obj_by_id(src); if (!src_obj){return};
        let dst_obj = find_obj_by_id(dst); if (!dst_obj){return};
        rayParams_new.sourceOffset =  src_obj.position;
        rayParams_new.destOffset =    dst_obj.position;
        rayParams_new.radius0 = size
        rayParams_new.radius1 = size/4.0
        if (isNaN(src_obj.position.x) || isNaN(src_obj.position.y)){return}
        // let lightningColor = new THREE.Color( 0xFFB0FF );
        let lightningStrike = new window.glb.import_LightningStrike( rayParams_new );
        let lightningMaterial = new THREE.MeshBasicMaterial( { color: color } );
        let lightningStrikeMesh = new THREE.Mesh( lightningStrike, lightningMaterial );
        window.glb.scene.add( lightningStrikeMesh );
        window.glb.flash_Obj.push({
            'create_time':new Date(),
            'dur':dur,
            'valid':true,
            'mesh':lightningStrikeMesh,
            'update_target':lightningStrike,
        })
    }
}
function parse_core_obj(str){
    const pattern = get_reg_exp(">>v2dx\\('(.*?)',([^,]*),([^,]*),([^,]*),(.*)\\)")
    let match_res = str.match(pattern)
    let name = match_res[1]

    // z --> y, y --- z reverse z axis and y axis
    let pos_x = parseFloat(match_res[2])
    let pos_y = parseFloat(match_res[4])
    let pos_z = -parseFloat(match_res[3])

    // z --> y, y --- z reverse z axis and y axis
    let ro_x = match_karg(str, 'ro_x', 'float', 0)
    let ro_y = match_karg(str, 'ro_z', 'float', 0)
    let ro_z = -match_karg(str, 'ro_y', 'float', 0)

    // pattern.test(str)
    let name_split = name.split('|')
    let type = name_split[0]
    let my_id = name_split[1]
    let color_str = name_split[2]
    let size = parseFloat(name_split[3])
    
    // swap Y and Z, e.g. 'XYZ' -> 'XZY'
    let ro_order = match_karg(str, 'ro_order', 'str', 'XYZ')
    ro_order = ro_order.replace('Y','T');ro_order = ro_order.replace('Z','Y');ro_order = ro_order.replace('T','Z');

    
    let parsed_obj_info = {};
    parsed_obj_info['name'] = name;
    parsed_obj_info['pos_x'] = pos_x;
    parsed_obj_info['pos_y'] = pos_y;
    parsed_obj_info['pos_z'] = pos_z;

    parsed_obj_info['ro_x'] = ro_x;
    parsed_obj_info['ro_y'] = ro_y;
    parsed_obj_info['ro_z'] = ro_z;

    parsed_obj_info['type'] = type;
    parsed_obj_info['my_id'] = my_id;
    parsed_obj_info['color_str'] = color_str;
    parsed_obj_info['size'] = size;
    parsed_obj_info['label_marking'] = match_karg(str, 'label', 'str', `id ${my_id}`);
    parsed_obj_info['label_color'] = match_karg(str, 'label_color', 'str', 'black');
    parsed_obj_info['label_size'] = match_karg(str, 'label_size', 'float', null);
    parsed_obj_info['label_offset'] = match_karg(str, 'label_offset', 'arr_float', null);
    parsed_obj_info['label_opacity'] = match_karg(str, 'label_opacity', 'float', 1);
    parsed_obj_info['opacity'] = match_karg(str, 'opacity', 'float', 1);
    parsed_obj_info['track_n_frame'] = match_karg(str, 'track_n_frame', 'int', 0);
    parsed_obj_info['renderOrder'] = match_karg(str, 'renderOrder', 'int', 0);
    parsed_obj_info['track_tension'] = match_karg(str, 'track_tension', 'float', 0);
    parsed_obj_info['track_color'] = match_karg(str, 'track_color', 'str', color_str);
    parsed_obj_info['ro_order'] = ro_order;
    parsed_obj_info['fade_step'] = match_karg(str, 'fade_step', 'int', null); 
    parsed_obj_info['label_bgcolor'] = match_karg(str, 'label_bgcolor', 'str', null); 
    
    // check parameters
    if (parsed_obj_info['fade_step'] && parsed_obj_info['fade_step']<=0){ alert('fade_step must >=1 !') }
    
    apply_update(parsed_obj_info)
}

