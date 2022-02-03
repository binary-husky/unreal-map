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

function parse_update_env(buf_str) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_env") != -1){
            parse_env(str)
        }
    }
}

function parse_init(buf_str) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_style") != -1){
            parse_style(str)
        }
        if(str.search(">>geometry_rotate_scale_translate") != -1){
            parse_geometry(str)
        }
        if(str.search(">>advanced_geometry_rotate_scale_translate") != -1){
            parse_advanced_geometry(str)
        }
    }
}

function parse_update_without_re(pp){
    let parsed_frame = window.glb.parsed_core_L[pp]
    for (let i = 0; i < parsed_frame.length; i++) {
        let parsed_obj_info = parsed_frame[i]
        let my_id = parsed_obj_info['my_id']
        // find core obj by my_id
        let object = find_obj_by_id(my_id)

        apply_update(object, parsed_obj_info)
    }
}




var init_terrain = false;

function parse_env(str){
    let re_style = />>set_env\('(.*)'/
    let re_res = str.match(re_style)
    let style = re_res[1]
    if(style=="terrain"){
        let get_theta = />>set_env\('terrain',theta=([^,)]*)/
        let get_theta_res = str.match(get_theta)
        let theta = parseFloat(get_theta_res[1])
        
        ////////////////////// add terrain /////////////////////
        let width = 30; let height = 30;
        let Segments = 200;
        if (!init_terrain){
            init_terrain=true;
        }else{
            window.glb.scene.remove(window.glb.terrain_mesh);
        }
        let geometry = new THREE.PlaneBufferGeometry(width, height, Segments - 1, Segments - 1); //(width, height,widthSegments,heightSegments)
        geometry.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
        window.glb.terrain_mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({}));
        window.glb.scene.add(window.glb.terrain_mesh);
        let array = geometry.attributes.position.array;
        for (let i = 0; i < Segments * Segments; i++) {
            let x = array[i * 3 + 0];
            let _x_ = array[i * 3 + 0];
            let z = array[i * 3 + 2];
            let _y_ = -array[i * 3 + 2];

            let A=0.05; 
            let B=0.2;
            let X_ = _x_*Math.cos(theta) + _y_*Math.sin(theta);
            let Y_ = -_x_*Math.sin(theta) + _y_*Math.cos(theta);
            let Z = -1 +B*( (0.1*X_) ** 2 + (0.1*Y_) ** 2 )- A * Math.cos(2 * Math.PI * (0.3*X_))  - A * Math.cos(2 * Math.PI * (0.5*Y_))
            Z = -Z;
            Z = (Z-1)*4;
            Z = Z - 0.1
            array[i * 3 + 1] = Z

        }
        geometry.computeBoundingSphere(); geometry.computeVertexNormals();
        console.log('update terrain')
    }
    if(style=="terrain_rm"){
        if (!init_terrain){
        }else{
            window.glb.scene.remove(window.glb.terrain_mesh);
        }
        
    }
    if(style=="clear_all"){
        if (!init_terrain){
        }else{
            window.glb.scene.remove(window.glb.terrain_mesh);
        }
        for (let i = window.glb.core_Obj.length-1; i>=0 ; i--) {
            window.glb.scene.remove(window.glb.core_Obj[i]);
            window.glb.core_Obj.pop();
        }
    }
}


function parse_style(str){
    //E.g. >>flash('lightning',src=0.00000000e+00,dst=1.00000000e+01,dur=1.00000000e+00)
    let re_style = />>set_style\('(.*)'/
    let re_res = str.match(re_style)
    let style = re_res[1]
    if(style=="terrain"){
        console.log('use set_env')
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
                if ((x*x+y*y+z*z)>20000){break;}
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
        window.glb.renderer.shadowMapEnabled	= true
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
    const pattern = />>advanced_geometry_rotate_scale_translate\('(.*)','(.*)',([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)(.*)\)/
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
    // very basic shapes
    window.glb.base_geometry[name] = eval('new THREE.'+build_cmd)
    window.glb.base_geometry[name] = geo_transform(window.glb.base_geometry[name], ro_x, ro_y, ro_z, scale_x, scale_y, scale_z, trans_x, trans_y, trans_z);
}


function parse_geometry(str){
    const pattern = />>geometry_rotate_scale_translate\('(.*)',([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)(.*)\)/
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
    let parsed_frame = []
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if (str.search(">>v2dx") != -1) {
            // name, xpos, ypos, zpos, dir=0, **kargs
            parse_core_obj(str, parsed_frame)
        }
    }
    window.glb.parsed_core_L[pp] = parsed_frame
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

function parse_line(str){
    const pattern = />>line3d\('(.*?)',(.*)\)/
    let match_res = str.match(pattern)
    let name = match_res[1] // norm|0|Violet|0.010

    let name_split = name.split('|')
    let type = name_split[0] // norm
    let my_id = parseInt(name_split[1]) // 0
    let color_str = name_split[2]   //Violet
    let size = parseFloat(name_split[3]) //0.010

    // x_arr
    let re_res = str.match(/x_arr=\[(.*?)\]/)
    let x_arr = re_res[1].split(',')
    for (i=0; i<x_arr.length; i++){
        x_arr[i] = parseFloat(x_arr[i])
    }

    // y_arr
    re_res = str.match(/z_arr=\[(.*?)\]/)
    let y_arr = re_res[1].split(',')
    for (i=0; i<y_arr.length; i++){
        y_arr[i] = parseFloat(y_arr[i])
    }

    // z_arr
    re_res = str.match(/y_arr=\[(.*?)\]/)
    let z_arr = re_res[1].split(',')
    for (i=0; i<z_arr.length; i++){
        z_arr[i] = -parseFloat(z_arr[i])
    }

    let res;
    let label_marking = `id ${my_id}`
    let label_color = 'black'
    // use label
    res = str.match(/label='(.*?)'/)
    // console.log(res)
    if (!(res === null)){
        label_marking = res[1]
    }

    res = str.match(/label_color='(.*?)'/)
    if (!(res === null)){
        label_color = res[1]
    }else{
        label_color = 'black'
    }

    let opacity_RE = str.match(/opacity=([^,)]*)/);
    let opacity = (!(opacity_RE === null))?parseFloat(opacity_RE[1]):1;

    let tension_RE = str.match(/tension=([^,)]*)/);
    let tension = (!(tension_RE === null))?parseFloat(tension_RE[1]):0;


    // find core obj by my_id
    let object = find_lineobj_by_id(my_id)
    let parsed_obj_info = {} 
    parsed_obj_info['name'] = name  
    parsed_obj_info['x_arr'] = x_arr  
    parsed_obj_info['y_arr'] = y_arr
    parsed_obj_info['z_arr'] = z_arr

    parsed_obj_info['type'] = type  
    parsed_obj_info['tension'] = tension
    parsed_obj_info['my_id'] = my_id  
    parsed_obj_info['color_str'] = color_str  
    parsed_obj_info['size'] = size
    parsed_obj_info['label_marking'] = label_marking
    parsed_obj_info['label_color'] = label_color
    parsed_obj_info['opacity'] = opacity
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

//修改颜色
function changeCoreObjColor(object, color_str){
    const colorjs = color_str;
    object.material.color.set(colorjs)
    object.color_str = color_str;
}

const ARC_SEGMENTS = 150;
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
        for ( let i = 0; i < ARC_SEGMENTS; i ++ ) {
            const t = i / ( ARC_SEGMENTS - 1 );
            curve.getPoint( t, point_tmp ); 
            positions_catmull.push( point_tmp.x, point_tmp.y, point_tmp.z)
        }
        //apply calculated positions_catmull
        curve.mesh_line.geometry.setPositions( positions_catmull );
        curve.mesh_line.computeLineDistances();
        //     positions.push( new THREE.Vector3(x_arr[i], y_arr[i], z_arr[i]) );
        // }
        // curve.points = positions
        // const position = curve.mesh.geometry.attributes.position;
        // const point = new THREE.Vector3();
        // for ( let i = 0; i < ARC_SEGMENTS; i ++ ) {
        //     const t = i / ( ARC_SEGMENTS - 1 );
        //     curve.getPoint( t, point );
        //     position.setXYZ( i, point.x, point.y, point.z );
        // }
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
        for ( let i = 0; i < ARC_SEGMENTS; i ++ ) {
            const t = i / ( ARC_SEGMENTS - 1 );
            curve.getPoint( t, point_tmp ); 
            positions_catmull.push( point_tmp.x, point_tmp.y, point_tmp.z)
        }
        // geo
        const geometry = new window.glb.import_LineGeometry();
        geometry.setPositions( positions_catmull );
        // material
        matLine = new window.glb.import_LineMaterial( {
            color: parsed_obj_info['color_str'],
            linewidth: parsed_obj_info['size'], // in world units with size attenuation, pixels otherwise
            // vertexColors: true,
            // alphaToCoverage: true,
            dashed: false,
            worldUnits: true,
            opacity: parsed_obj_info['opacity'],
            transparent: (parsed_obj_info['opacity']!=1)
        } );


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
        rayParams_new.sourceOffset =  find_obj_by_id(src).position;
        rayParams_new.destOffset =    find_obj_by_id(dst).position;
        rayParams_new.radius0 = size
        rayParams_new.radius1 = size/4.0
        if (isNaN(find_obj_by_id(src).position.x) || isNaN(find_obj_by_id(src).position.y)){return}
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
        rayParams_new.sourceOffset =  find_obj_by_id(src).position;
        rayParams_new.destOffset =    find_obj_by_id(dst).position;
        rayParams_new.radius0 = size
        rayParams_new.radius1 = size/4.0
        if (isNaN(find_obj_by_id(src).position.x) || isNaN(find_obj_by_id(src).position.y)){return}
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
function parse_core_obj(str, parsed_frame){
    // ">>v2dx(x, y, dir, xxx)"
    // each_line[i].replace('>>v2dx(')
    // ">>v2dx('ball|8|blue|0.05',1.98948879e+00,-3.15929300e+00,-4.37260984e-01,ro_x=0,ro_y=0,ro_z=2.10134351e+00,label='',label_color='white',attack_range=0)"
    const pattern = />>v2dx\('(.*?)',([^,]*),([^,]*),([^,]*),(.*)\)/
    let match_res = str.match(pattern)
    let name = match_res[1]

    let pos_x = parseFloat(match_res[2])
    // z --> y, y --- z reverse z axis and y axis
    let pos_y = parseFloat(match_res[4])
    // z --> y, y --- z reverse z axis and y axis
    let pos_z = -parseFloat(match_res[3])

    let ro_x_RE = str.match(/ro_x=([^,)]*)/);
    let ro_x = (!(ro_x_RE === null))?parseFloat(ro_x_RE[1]):0;
    // z --> y, y --- z reverse z axis and y axis
    let ro_z_RE = str.match(/ro_z=([^,)]*)/);
    let ro_y = (!(ro_z_RE === null))?parseFloat(ro_z_RE[1]):0;
    // z --> y, y --- z reverse z axis and y axis
    let ro_y_RE = str.match(/ro_y=([^,)]*)/);
    let ro_z = (!(ro_y_RE === null))?-parseFloat(ro_y_RE[1]):0;

    // pattern.test(str)
    let name_split = name.split('|')
    let type = name_split[0]
    let my_id = parseInt(name_split[1])
    let color_str = name_split[2]
    let size = parseFloat(name_split[3])
    let label_marking = `id ${my_id}`
    let label_color = "black"
    // find hp 
    const hp_pattern = /health=([^,)]*)/
    let hp_match_res = str.match(hp_pattern)
    if (!(hp_match_res === null)){
        let hp = parseFloat(hp_match_res[1])
        if (Number(hp) === hp && hp % 1 === 0){
            // is an int
            hp = Number(hp);
        }
        else{
            hp = hp.toFixed(2);
        }
        label_marking = `HP ${hp}`
    }
    // e.g. >>v2dx('tank|12|b|0.1',-8.09016994e+00,5.87785252e+00,0,vel_dir=0,health=0,label='12',attack_range=0)
    let res;
    // use label
    res = str.match(/label='(.*?)'/)
    // console.log(res)
    if (!(res === null)){
        label_marking = res[1]
    }

    res = str.match(/label_color='(.*?)'/)
    if (!(res === null)){
        label_color = res[1]
    }else{
        label_color = 'black'
    }

    let opacity_RE = str.match(/opacity=([^,)]*)/);
    let opacity = (!(opacity_RE === null))?parseFloat(opacity_RE[1]):1;

    let track_n_frame_RE = str.match(/track_n_frame=([^,)]*)/);
    let track_n_frame = (!(track_n_frame_RE === null))?parseInt(track_n_frame_RE[1]):0;

    let track_tension_RE = str.match(/track_tension=([^,)]*)/);
    let track_tension = (!(track_tension_RE === null))?parseFloat(track_tension_RE[1]):0;

    let track_color_RE = str.match(/track_color='(.*?)'/)
    let track_color = (!(track_color_RE === null))?track_color_RE[1]:color_str

    // find core obj by my_id
    let object = find_obj_by_id(my_id)
    let parsed_obj_info = {} 
    parsed_obj_info['name'] = name  
    parsed_obj_info['pos_x'] = pos_x  
    parsed_obj_info['pos_y'] = pos_y
    parsed_obj_info['pos_z'] = pos_z

    parsed_obj_info['ro_x'] = ro_x  
    parsed_obj_info['ro_y'] = ro_y
    parsed_obj_info['ro_z'] = ro_z

    parsed_obj_info['type'] = type  
    parsed_obj_info['my_id'] = my_id  
    parsed_obj_info['color_str'] = color_str  
    parsed_obj_info['size'] = size  
    parsed_obj_info['label_marking'] = label_marking
    parsed_obj_info['label_color'] = label_color
    parsed_obj_info['opacity'] = opacity
    parsed_obj_info['track_n_frame'] = track_n_frame
    parsed_obj_info['track_tension'] = track_tension
    parsed_obj_info['track_color'] = track_color
    
    apply_update(object, parsed_obj_info)
    parsed_frame.push(parsed_obj_info)
}

function find_obj_by_id(my_id){
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        if (window.glb.core_Obj[i].my_id == my_id) {
            return window.glb.core_Obj[i];
        }
    }
    return null
}
