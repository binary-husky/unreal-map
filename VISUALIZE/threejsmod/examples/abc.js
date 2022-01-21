// import * as THREE from '/build/three.module.js';
// add new property 'opacity'
// (1) parse (@parse_core_obj)
// (2) apply_update.新增对象 (@apply_update, abc_obs.js)
// (3) addCoreObj (@addCoreObj, abc_obs.js) create next
// (4) apply_update.已经创建了对象 (@apply_update, abc_obs.js)
// (5) force_move_all. (@apply_update, abc.js)
// (6) change_position_rotation_size. (abc.js)

import Stats from '/examples/jsm/libs/stats.module.js';
import { GUI } from '/examples/jsm/libs/dat.gui.module.js';
import { LightningStrike } from '/examples/jsm/geometries/LightningStrike.js';
import { OrbitControls } from '/examples/jsm/controls/OrbitControls.js';

window.glb = Object()

window.glb.clock = new THREE.Clock();
window.glb.scene = null;
window.glb.buf_storage = ''
window.glb.BarFolder=null;
window.glb.camera=null;
let container, stats;
var renderer;
window.glb.controls=null;
// var window.glb.scene;
// 历史轨迹
window.glb.core_L = [];
window.glb.parsed_core_L = []
window.glb.core_Obj = [];
window.glb.flash_Obj = [];
window.glb.base_geometry = {};
// global vars
var play_fps = 10;
var dt_since = 0;
var dt_threshold = 1 / play_fps;
var play_pointer = 0;
var buf_str = '';
var DEBUG = false;
var transfer_ongoing = false;
// GUI
var request=null;
var req_interval=2.0;
let panelSettings = {
        'play fps':play_fps,
        'play pointer':0,
        'data req interval': req_interval,
        'reset to read new': null,
        'pause': null,
        'next frame': null,
        'previous frame': null,
};
// color dictionary 
var color_dic = {
    k:0x000000,
    r:0xff0000, 
    g:0x00ff00, 
    b:0x0000ff, 
};
var color_layer = {
    k:0,
    r:1, 
    g:2, 
    b:3, 
};

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



//////////////////////main read function//////////////////////////
var coreReadFunc = function (auto_next=true) {
    if (transfer_ongoing){
        console.log('ongoing')
        req_interval = req_interval+1;
        panelSettings['data req interval'] = req_interval
        if (!DEBUG){setTimeout(coreReadFunc, req_interval*1000);}
        return
    }
    request = new XMLHttpRequest();
    request.open('POST', `/up`);
    request.overrideMimeType("text/plain; charset=x-user-defined");
    request.timeout = 30*1000; // 30sec 超时时间，单位是毫秒
    request.ontimeout = function (e) {
        transfer_ongoing = false;
        alert('xhr request timeout!')
    };
    request.onload = () => {
        console.log('loaded')
        let inflat;
        if (!DEBUG){
            let byteArray = toUint8Arr(request.response);
            inflat = pako.inflate(byteArray,{to: 'string'}); //window.pako.inflate(byteArray)
        }else{
            inflat = request.responseText
        }
        let n_new_data = core_update(inflat)
        if (n_new_data==0){
            req_interval = req_interval+1;
            if (req_interval>100){
                req_interval=100;
            }
            panelSettings['data req interval'] = req_interval
        }else{
            req_interval = 0
            panelSettings['data req interval'] = req_interval
        }

        transfer_ongoing = false;
    };
    request.send();
    console.log('send req')
    transfer_ongoing = true;
    if(auto_next && !DEBUG){setTimeout(coreReadFunc, req_interval*1000);}
    console.log('next update '+req_interval)
}
panelSettings['reset to read new'] = function (){
    if (transfer_ongoing){
        request.abort()
        transfer_ongoing = false;
    }
    coreReadFunc(false)
}
setTimeout(coreReadFunc, 100);

panelSettings['pause'] = function (){
    if (panelSettings['play fps']>0){
        window.glb.prev_fps = panelSettings['play fps']
        panelSettings['play fps'] = 0
        play_fps = panelSettings['play fps'];
        dt_since = 0; dt_threshold = 1 / play_fps;
    }else{
        panelSettings['play fps'] = window.glb.prev_fps
        play_fps = panelSettings['play fps'];
        dt_since = 0; dt_threshold = 1 / play_fps;
    }
}
panelSettings['next frame'] = function (){
    if (play_pointer >= window.glb.core_L.length) {play_pointer = 0;}
    else{play_pointer = play_pointer + 1;}
    
    panelSettings['play pointer'] = play_pointer;
    if(play_fps==0){force_move_all(play_pointer)}
}
panelSettings['previous frame'] = function (){
    if(play_pointer>0) {
        play_pointer = play_pointer - 1;
        panelSettings['play pointer'] = play_pointer;
        if(play_fps==0){force_move_all(play_pointer)}
    }
}
////////////////////////////////////////////////////////////


function removeEntity(object) {
    var selectedObject = window.glb.scene.getObjectByName(object.name);
    window.glb.scene.remove(selectedObject);
}



function parse_update_without_re(play_pointer){
    let parsed_frame = window.glb.parsed_core_L[play_pointer]
    for (let i = 0; i < parsed_frame.length; i++) {
        let parsed_obj_info = parsed_frame[i]
        let my_id = parsed_obj_info['my_id']
        // find core obj by my_id
        let object = find_obj_by_id(my_id)

        apply_update(object, parsed_obj_info)
    }
}






function parse_core_obj(str, parsed_frame){
    // ">>v2dx(x, y, dir, xxx)"
    // each_line[i].replace('>>v2dx(')
    // ">>v2dx('ball|8|blue|0.05',1.98948879e+00,-3.15929300e+00,-4.37260984e-01,ro_x=0,ro_y=0,ro_z=2.10134351e+00,label='',label_color='white',attack_range=0)"
    const pattern = />>v2dx\('(.*)',([^,]*),([^,]*),([^,]*),(.*)\)/
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

    apply_update(object, parsed_obj_info)
    parsed_frame.push(parsed_obj_info)
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


function find_obj_by_id(my_id){
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        if (window.glb.core_Obj[i].my_id == my_id) {
            return window.glb.core_Obj[i];
        }
    }
    return null
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
        let lightningStrike = new LightningStrike( rayParams_new );
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
        let lightningStrike = new LightningStrike( rayParams_new );
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

let currentTime = 0;
function check_flash_life_cyc(delta_time){
    // delta_time: seconds
    currentTime += delta_time
    for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
        let nowTime = new Date();
        let dTime = nowTime - window.glb.flash_Obj[i]['create_time']   // 相差的毫秒数
        if (dTime>=window.glb.flash_Obj[i]['dur']*1000){
            window.glb.flash_Obj[i]['valid'] = false;
            window.glb.scene.remove(window.glb.flash_Obj[i]['mesh']);
        }
    }
    window.glb.flash_Obj = window.glb.flash_Obj.filter(function (s) {return s['valid'];});
    for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
        window.glb.flash_Obj[i]['update_target'].update(currentTime)
    }
}

function parse_update_core(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    let parsed_frame = []
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if (str.search(">>v2dx") != -1) {
            // name, xpos, ypos, zpos, dir=0, **kargs
            parse_core_obj(str, parsed_frame)
        }
    }
    window.glb.parsed_core_L[play_pointer] = parsed_frame
}

function parse_update_flash(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if(str.search(">>flash") != -1){
            parse_flash(str)
        }
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

function parse_update_env(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_env") != -1){
            parse_env(str)
        }
    }
}

function parse_init(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        let str = each_line[i]
        if(str.search(">>set_style") != -1){
            parse_style(str)
        }
        if(str.search(">>geometry_rotate_scale_translate") != -1){
            parse_geometry(str)
        }
    }
}

function parse_time_step(play_pointer){
    if(window.glb.parsed_core_L[play_pointer]) {
        buf_str = window.glb.core_L[play_pointer]
        parse_update_env(buf_str, play_pointer)
        parse_update_without_re(play_pointer)
        parse_update_flash(buf_str, play_pointer)
    }else{
        buf_str = window.glb.core_L[play_pointer]
        parse_init(buf_str, play_pointer)
        parse_update_env(buf_str, play_pointer)
        parse_update_core(buf_str, play_pointer)
        parse_update_flash(buf_str, play_pointer)
    }
}


// warning: python mod operator is different from js mod operator
function reg_rad(rad){
    let a = (rad + Math.PI) 
    let b = (2 * Math.PI)

    return ((a%b)+b) % b - Math.PI
}

function reg_rad_at(rad, ref){
    return reg_rad(rad-ref) + ref
}

// ConeGeometry 锥体， radius — 圆锥底部的半径，默认值为1。height — 圆锥的高度，默认值为1。，
// CylinderGeometry 圆柱体 radiusTop — 圆柱的顶部半径，默认值是1。 radiusBottom — 圆柱的底部半径，默认值是1。 height — 圆柱的高度，默认值是1。
// SphereGeometry 球体 radius — 球体半径，默认为1

function change_position_rotation_size(object, percent){
    object.position.x = object.prev_pos.x * (1 - percent) + object.next_pos.x * percent
    object.position.y = object.prev_pos.y * (1 - percent) + object.next_pos.y * percent
    object.position.z = object.prev_pos.z * (1 - percent) + object.next_pos.z * percent

    let reg__next_ro_x = reg_rad_at(object.next_ro.x, object.prev_ro.x)
    let reg__next_ro_y = reg_rad_at(object.next_ro.y, object.prev_ro.y)
    let reg__next_ro_z = reg_rad_at(object.next_ro.z, object.prev_ro.z)
    object.rotation.x = object.prev_ro.x * (1 - percent) + reg__next_ro_x * percent
    object.rotation.y = object.prev_ro.y * (1 - percent) + reg__next_ro_y * percent
    object.rotation.z = object.prev_ro.z * (1 - percent) + reg__next_ro_z * percent

    let size = object.prev_size * (1 - percent)  + object.next_size * percent
    let opacity = object.prev_opacity * (1 - percent)  + object.next_opacity * percent
    if (object.material.opacity!=opacity){ object.material.opacity=opacity }
    changeCoreObjSize(object, size)
}
function force_move_all(play_pointer){ // 手动调整进度条时触发
    parse_time_step(play_pointer)
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        object.prev_pos.x = object.next_pos.x; object.prev_pos.y = object.next_pos.y; object.prev_pos.z = object.next_pos.z
        object.prev_ro.x = object.next_ro.x; object.prev_ro.y = object.next_ro.y; object.prev_ro.z = object.next_ro.z
        object.prev_opacity = object.next_opacity
        change_position_rotation_size(object, 1)
    }	
}
function move_to_future(percent) {
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        change_position_rotation_size(object, percent)
    }
}

function set_next_play_frame() {
    if (window.glb.core_L.length == 0) { return; }
    if (play_pointer >= window.glb.core_L.length) {play_pointer = 0;}
    parse_time_step(play_pointer)
    play_pointer = play_pointer + 1
    if (play_pointer >= window.glb.core_L.length) {play_pointer = 0;}
    panelSettings['play pointer'] = play_pointer;
}

function onWindowResize() {
    window.glb.camera.aspect = window.innerWidth / window.innerHeight;
    window.glb.camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    render();
    stats.update();
}

function render() {
    const delta = window.glb.clock.getDelta();
    dt_since = dt_since + delta;
    if (dt_since > dt_threshold) {
        dt_since = 0;
        set_next_play_frame();
        move_to_future(0);
    }
    else {
        let percent = Math.min(dt_since / dt_threshold, 1.0);
        move_to_future(percent);
    }
    check_flash_life_cyc(delta)
    renderer.render(window.glb.scene, window.glb.camera);
}


function init() {
    container = document.createElement('div');
    document.body.appendChild(container);
    // 透视相机  Fov, Aspect, Near, Far – 相机视锥体的远平面
    window.glb.camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, 0.001, 10000);
    // window.glb.camera.up.set(0,0,1);一个 0 - 31 的整数 Layers 对象为 Object3D 分配 1个到 32 个图层。32个图层从 0 到 31 编号标记。
    window.glb.camera.layers.enable(0); // 启动0图层
    window.glb.scene = new THREE.Scene();

    const grid = new THREE.GridHelper( 500, 500, 0xffffff, 0x555555 );
    grid.position.y = 0
    grid.visible = false
    window.glb.scene.add(grid);
    const light = new THREE.PointLight(0xffffff, 1);
    light.layers.enable(0); // 启动0图层

    window.glb.scene.add(window.glb.camera);
    window.glb.camera.add(light);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);


    stats = new Stats();
    container.appendChild(stats.dom);

    window.glb.controls = new OrbitControls(window.glb.camera, renderer.domElement);
    // window.glb.controls.object.up = new THREE.Vector3( 1, 0, 0 )
    window.glb.controls.target.set(0, 0, 0); // 旋转的焦点在哪0,0,0即原点
    window.glb.camera.position.set(0, 50, 0)
    window.glb.controls.update();
    window.glb.controls.autoRotate = false;

    
    const panel = new GUI( { width: 310 } );
    const Folder1 = panel.addFolder( 'General' );
    // FPS adjust
    Folder1.add(panelSettings, 'play fps', 0, 144, 1).listen().onChange(
        function change_fps(fps) {
            play_fps = fps;
            dt_since = 0;
            dt_threshold = 1 / play_fps;
        });
    Folder1.add(panelSettings, 'data req interval', 1, 100, 1).listen().onChange(
        function (interval) {
            req_interval = interval;
    });
    Folder1.add( panelSettings, 'reset to read new' );
    Folder1.open();

    window.glb.BarFolder = panel.addFolder('Play Pointer');
    window.glb.BarFolder.add(panelSettings, 'play pointer', 0, 10000, 1).listen().onChange(
        function (p) {
            play_pointer = p;
            if(play_fps==0){
                force_move_all(play_pointer)
            }
    });
    window.glb.BarFolder.add( panelSettings, 'pause'          );
    window.glb.BarFolder.add( panelSettings, 'next frame'     );
    window.glb.BarFolder.add( panelSettings, 'previous frame' );
    window.glb.BarFolder.open();


    
    // window.glb.terrain_mesh.scale.y = 50.0;
    // this.mesh.position.x = this.width / 2;
    // this.mesh.position.z = this.height / 2;
    



    window.addEventListener('resize', onWindowResize);
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
        geometry.applyMatrix(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
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
    }else if(style=="nogrid"){
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
        renderer.shadowMapEnabled	= true
    }
}

init();
animate();