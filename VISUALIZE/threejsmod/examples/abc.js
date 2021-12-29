// import * as THREE from '/build/three.module.js';
import Stats from '/examples/jsm/libs/stats.module.js';
import { GUI } from '/examples/jsm/libs/dat.gui.module.js';
import { LightningStrike } from '/examples/jsm/geometries/LightningStrike.js';

// import { FlyControls } from '/examples/jsm/controls/FlyControls.js';
import { OrbitControls } from '/examples/jsm/controls/OrbitControls.js';
let container, stats;
var camera, renderer;
var controls;
var scene;
// let theta = 0;
// const radius = 100;
const clock = new THREE.Clock();
// 历史轨迹
var buf_storage = ''
var core_L = [];
var core_Obj = [];
var flash_Obj = [];
// global vars
var play_fps = 10;
var dt_since = 0;
var dt_threshold = 1 / play_fps;
var play_pointer = 0;
var buf_str = '';
var init_cam = false;
var DEBUG = false;
var transfer_ongoing = false;
// GUI
var request=null;
var req_interval=2.0;
var BarFolder;
let panelSettings = {
        'play fps':play_fps,
        'play pointer':0,
        'data req interval': req_interval,
        'reset to read new': null
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

function toUint8Arr(str) {
    const buffer = [];
    for (let i = 0; i < str.length; i++) {
        buffer.push(str.charCodeAt(i)&0xff);
    }
    return Uint8Array.from(buffer);
}

function core_update(buf) {
    const EOFF = '>>v2d_show()\n'
    const EOFX = '>>v2d_init()\n'
    let tmp = buf.split(EOFF);

    // filter empty
    tmp = tmp.filter(function (s) {
        return s&&s.trim(); // '' --> exclude,  false--> exclude, true --> include
    });
    if (tmp.length==0){return 0;}


    if (buf_storage){
        tmp[0] = buf_storage + tmp[0]
    }
    buf_storage = ''
    if (!buf.endsWith(EOFF)){
        buf_storage = tmp.pop()
    }
    // if (tmp.length==0){return 0;}
    //check EOFX
    let eofx_i = -1;
    for (let i = 0; i < tmp.length; i++) {
        if (tmp[i].indexOf(EOFX) != -1){
            eofx_i = i
        }
    }
    if (eofx_i>=0){
        alert('new session detected')
        core_L = []
        parsed_core_L = []
        for (let i = core_Obj.length-1; i>=0; i--) {
            scene.remove(core_Obj[i]);
        }
        core_Obj = []
        tmp.splice(0, eofx_i);
    }
    //
    core_L = core_L.concat(tmp);
    if (core_L.length>1e8){
        core_L.splice(0, tmp.length);
        play_pointer = play_pointer - tmp.length;
        if (play_pointer<0){
            play_pointer=0;
        }
    }
    BarFolder.__controllers[0].max(core_L.length);
    return tmp.length;
}

////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////


function addCoreObj(my_id, color_str, geometry, x, y, z, geometry_size, currentSize, label_marking, label_color){
    const object = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: color_str }));
    object.my_id = my_id;
    object.color_str = color_str;
    object.position.x = x; 
    object.position.y = y; 
    object.position.z = z; 
    object.next_pos = Object.create(object.position);
    object.prev_pos = Object.create(object.position);
    
    object.rotation.x = 0;  
    object.rotation.y = 0;  
    object.rotation.z = 0;  
    object.initialSize = geometry_size
    object.currentSize = currentSize
    object.generalSize = geometry_size
    object.prev_size = currentSize
    object.next_size = currentSize

    object.scale.x = 1; 
    object.scale.y = 1; 
    object.scale.z = 1; 
    object.label_marking = label_marking
    object.label_color = label_color

    if (!init_cam){
        console.log('first')
        init_cam=true;
        controls.target.set(object.position.x, -geometry_size, object.position.z); // 旋转的焦点在哪0,0,0即原点
        camera.position.set(object.position.x, geometry_size*100, object.position.z)
    }
    if (label_marking){
        makeClearText(object, object.label_marking, object.label_color)
    }
    scene.add(object);
    core_Obj.push(object)
}

function makeClearText(object, text, textcolor, HWRatio=10){
    let textTextureHeight = 512
    let textTextureWidth = 512*HWRatio
    object.dynamicTexture  = new THREEx.DynamicTexture(textTextureWidth, textTextureHeight)
    let px = 256
    object.dynamicTexture.context.font	= `bolder ${px}px Verdana`
    object.dynamicTexture.drawText(text, 0, +80, textcolor)	// text, x ,y, fillStyle（font color）, contextFont
    const materialB = new THREE.SpriteMaterial({ map:  object.dynamicTexture.texture, depthWrite: false });
    const sprite = new THREE.Sprite(materialB);

    sprite.scale.x = 17* object.generalSize
    sprite.scale.y = 17* object.generalSize/HWRatio
    sprite.scale.z = 17* object.generalSize
    sprite.position.set(object.generalSize, object.generalSize, -object.generalSize);
    object.add(sprite)
}

function removeEntity(object) {
    var selectedObject = scene.getObjectByName(object.name);
    scene.remove(selectedObject);
}

function changeCoreObjColor(object, color_str){
    const colorjs = color_str;
    object.material.color.set(colorjs)
    object.color_str = color_str;
}

function changeCoreObjSize(object, size){
    let ratio_ = (size/object.initialSize)
    // console.log(size+'---'+object.initialSize+'---'+ratio_)
    // console.log(object.scale)
    object.scale.x = ratio_
    object.scale.y = ratio_
    object.scale.z = ratio_
    // console.log(object.scale)
    object.currentSize = size
}
var parsed_core_L = []
function parse_update_without_re(play_pointer){
    let parsed_frame = parsed_core_L[play_pointer]
    for (let i = 0; i < parsed_frame.length; i++) {
        let parsed_obj_info = parsed_frame[i]
        let my_id = parsed_obj_info['my_id']
        // find core obj by my_id
        let object = null
        for (let i = 0; i < core_Obj.length; i++) {
            if (core_Obj[i].my_id == my_id) {
                object = core_Obj[i];
                break;
            }
        }
        apply_update(object, parsed_obj_info)
    }
}



function choose_geometry(type, geometry_size){
    if (type=='tank' || type=='box'){
        return new THREE.BoxGeometry(geometry_size, geometry_size, geometry_size);
    }else if(type=='sphe' || type=='ball'){
        return new THREE.SphereGeometry(geometry_size);
    }else if(type=='cone'){
        return new THREE.ConeGeometry(geometry_size, 2*geometry_size);
    }else{
        return new THREE.SphereGeometry(geometry_size); // default
    }
}

function apply_update(object, parsed_obj_info){
    let name = parsed_obj_info['name']
    let pos_x = parsed_obj_info['pos_x']
    let pos_y = parsed_obj_info['pos_y']
    let pos_z = parsed_obj_info['pos_z']
    let type = parsed_obj_info['type']
    let my_id = parsed_obj_info['my_id']
    let color_str = parsed_obj_info['color_str']
    let size = parsed_obj_info['size']
    let label_marking = parsed_obj_info['label_marking']
    let label_color = parsed_obj_info['label_color']
    // 已经创建了对象
    if (object) {
        object.prev_pos = Object.assign({}, object.next_pos);
        object.prev_size = object.next_size;
        object.next_pos.x = pos_x; // -400 ~ 400
        object.next_pos.y = pos_y; // -400 ~ 400
        object.next_pos.z = pos_z; // -400 ~ 400
        object.next_size = size; // -400 ~ 400
        if (color_str != object.color_str) {
            changeCoreObjColor(object, color_str)
        }
        if (label_marking != object.label_marking || label_color !=object.label_color) {
            object.label_marking = label_marking
            object.label_color = label_color
            if (!object.dynamicTexture) {
                makeClearText(object, object.label_marking, object.label_color)
            }
            object.dynamicTexture.clear().drawText(label_marking, 0, +80, object.label_color)
        }
    }
    else {
        // create obj
        let currentSize = size;
        // geometry_size should never be 0
        let geometry_size;
        if (size == 0) {geometry_size = 0.1} else {geometry_size = currentSize}
        let geometry = choose_geometry(type, geometry_size);
        //function (my_id, color_str, geometry, x, y, z, size, label_marking){
        addCoreObj(my_id, color_str, geometry, pos_x, pos_y, pos_z, geometry_size, currentSize, label_marking, label_color)
    }
}

function parse_core_obj(str, core_Obj, parsed_frame){
    // ">>v2dx(x, y, dir, xxx)"
    // each_line[i].replace('>>v2dx(')
    const pattern = />>v2dx\('(.*)',([^,]*),([^,]*),([^,]*),(.*)\)/
    let match_res = str.match(pattern)
    let name = match_res[1]
    // z --> y, y --- z
    let pos_x = parseFloat(match_res[2])
    let pos_y = parseFloat(match_res[4])
    let pos_z = parseFloat(match_res[3])
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

    // find core obj by my_id
    let object = null
    for (let i = 0; i < core_Obj.length; i++) {
        if (core_Obj[i].my_id == my_id) {
            object = core_Obj[i];
            break;
        }
    }
    let parsed_obj_info = {} 
    parsed_obj_info['name'] = name  
    parsed_obj_info['pos_x'] = pos_x  
    parsed_obj_info['pos_y'] = pos_y  
    parsed_obj_info['pos_z'] = pos_z  
    parsed_obj_info['type'] = type  
    parsed_obj_info['my_id'] = my_id  
    parsed_obj_info['color_str'] = color_str  
    parsed_obj_info['size'] = size  
    parsed_obj_info['label_marking'] = label_marking
    parsed_obj_info['label_color'] = label_color

    apply_update(object, parsed_obj_info)
    parsed_frame.push(parsed_obj_info)
}


function parse_flash(str, flash_Obj){
    //E.g. >>flash('lightning',src=0.00000000e+00,dst=1.00000000e+01,dur=1.00000000e+00)
    let re_type = />>flash\('(.*)'/
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
    make_flash(type, src, dst, dur, size, flash_Obj)
}


function find_obj_by_id(my_id){
    for (let i = 0; i < core_Obj.length; i++) {
        if (core_Obj[i].my_id == my_id) {
            return core_Obj[i];
        }
    }
    return null
}

function make_flash(type, src, dst, dur, size, flash_Obj){
    if (type=='lightning'){
        let rayParams_new = Object.create(rayParams_lightning);
        rayParams_new.sourceOffset =  find_obj_by_id(src).position;
        rayParams_new.destOffset =    find_obj_by_id(dst).position;
        rayParams_new.radius0 = size
        rayParams_new.radius1 = size/4.0
        if (isNaN(find_obj_by_id(src).position.x) || isNaN(find_obj_by_id(src).position.y)){return}
        let lightningColor = new THREE.Color( 0xFFB0FF );
        let lightningStrike = new LightningStrike( rayParams_new );
        let lightningMaterial = new THREE.MeshBasicMaterial( { color: lightningColor } );
        let lightningStrikeMesh = new THREE.Mesh( lightningStrike, lightningMaterial );
        scene.add( lightningStrikeMesh );
        flash_Obj.push({
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
        let lightningColor = new THREE.Color( 0xFFB0FF );
        let lightningStrike = new LightningStrike( rayParams_new );
        let lightningMaterial = new THREE.MeshBasicMaterial( { color: lightningColor } );
        let lightningStrikeMesh = new THREE.Mesh( lightningStrike, lightningMaterial );
        scene.add( lightningStrikeMesh );
        flash_Obj.push({
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
    for (let i = flash_Obj.length-1; i>=0; i--) {
        let nowTime = new Date();
        let dTime = nowTime - flash_Obj[i]['create_time']   // 相差的毫秒数
        if (dTime>=flash_Obj[i]['dur']*1000){
            flash_Obj[i]['valid'] = false;
            scene.remove(flash_Obj[i]['mesh']);
        }
    }
    flash_Obj = flash_Obj.filter(function (s) {return s['valid'];});
    for (let i = flash_Obj.length-1; i>=0; i--) {
        flash_Obj[i]['update_target'].update(currentTime)
    }
}

function parse_update_core(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    let parsed_frame = []
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if (str.search(">>v2dx") != -1) {
            // name, xpos, ypos, zpos, dir=0, **kargs
            parse_core_obj(str, core_Obj, parsed_frame)
        }
    }
    parsed_core_L[play_pointer] = parsed_frame
}

function parse_update_flash(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if(str.search(">>flash") != -1){
            parse_flash(str, flash_Obj)
        }
    }
}

function parse_update_style(buf_str, play_pointer) {
    let each_line = buf_str.split('\n')
    for (let i = 0; i < each_line.length; i++) {
        var str = each_line[i]
        if(str.search(">>set_style") != -1){
            parse_style(str)
        }
    }
}

function parse_time_step(play_pointer){
    if(parsed_core_L[play_pointer]) {
        buf_str = core_L[play_pointer]
        parse_update_without_re(play_pointer)
        parse_update_flash(buf_str, play_pointer)
    }else{
        buf_str = core_L[play_pointer]
        parse_update_core(buf_str, play_pointer)
        parse_update_flash(buf_str, play_pointer)
        parse_update_style(buf_str, play_pointer)
    }
}

// ConeGeometry 锥体， radius — 圆锥底部的半径，默认值为1。height — 圆锥的高度，默认值为1。，
// CylinderGeometry 圆柱体 radiusTop — 圆柱的顶部半径，默认值是1。 radiusBottom — 圆柱的底部半径，默认值是1。 height — 圆柱的高度，默认值是1。
// SphereGeometry 球体 radius — 球体半径，默认为1
function force_move_all(play_pointer){
    parse_time_step(play_pointer)
    for (let i = 0; i < core_Obj.length; i++) {
        let object = core_Obj[i]
        object.prev_pos.x = object.next_pos.x
        object.prev_pos.y = object.next_pos.y
        object.prev_pos.z = object.next_pos.z
        object.position.x = object.prev_pos.x * (1 - percent) + object.next_pos.x * percent
        object.position.y = object.prev_pos.y * (1 - percent) + object.next_pos.y * percent
        object.position.z = object.prev_pos.z * (1 - percent) + object.next_pos.z * percent
        let size = object.prev_size * (1 - percent)  + object.next_size * percent
        changeCoreObjSize(object, size)
    }	
}
function move_to_future(percent) {
    for (let i = 0; i < core_Obj.length; i++) {
        let object = core_Obj[i]
        object.position.x = object.prev_pos.x * (1 - percent) + object.next_pos.x * percent
        object.position.y = object.prev_pos.y * (1 - percent) + object.next_pos.y * percent
        object.position.z = object.prev_pos.z * (1 - percent) + object.next_pos.z * percent
        let size = object.prev_size * (1 - percent)  + object.next_size * percent
        changeCoreObjSize(object, size)
    }
}

function set_next_play_frame() {
    if (core_L.length == 0) { return; }
    if (play_pointer >= core_L.length) {play_pointer = 0;}
    parse_time_step(play_pointer)
    play_pointer = play_pointer + 1
    if (play_pointer >= core_L.length) {play_pointer = 0;}
    panelSettings['play pointer'] = play_pointer;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    render();
    stats.update();
}

function render() {
    const delta = clock.getDelta();
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
    renderer.render(scene, camera);
}


function init() {
    container = document.createElement('div');
    document.body.appendChild(container);
    // 透视相机  Fov, Aspect, Near, Far – 相机视锥体的远平面
    camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, 0.001, 10000);
    // camera.up.set(0,0,1);一个 0 - 31 的整数 Layers 对象为 Object3D 分配 1个到 32 个图层。32个图层从 0 到 31 编号标记。
    camera.layers.enable(0); // 启动0图层
    scene = new THREE.Scene();
    // scene.background = new THREE.Color(0xa0a0a0); //?
    // ground
    // const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1000, 1000), 
    // new THREE.MeshPhongMaterial({ color: 0x999999, depthWrite: false })
    // );
    // mesh.rotation.x = - Math.PI / 2;
    // mesh.receiveShadow = true;
    // scene.add(mesh);
    const grid = new THREE.GridHelper( 500, 500, 0xffffff, 0x555555 );
    grid.position.y = 0
    grid.visible = false
    scene.add(grid);
    const light = new THREE.PointLight(0xffffff, 1);
    light.layers.enable(0); // 启动0图层

    scene.add(camera);
    camera.add(light);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    //







    stats = new Stats();
    container.appendChild(stats.dom);

    controls = new OrbitControls(camera, renderer.domElement);
    // controls.object.up = new THREE.Vector3( 1, 0, 0 )
    controls.target.set(0, 0, 0); // 旋转的焦点在哪0,0,0即原点
    camera.position.set(0, 50, 0)
    controls.update();
    controls.autoRotate = false;

    
    const panel = new GUI( { width: 310 } );
    const Folder1 = panel.addFolder( 'General' );
    // FPS adjust
    Folder1.add(panelSettings, 'play fps', 0, 144, 1).onChange(
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

    BarFolder = panel.addFolder('Play Pointer');
    BarFolder.add(panelSettings, 'play pointer', 0, 10000, 1).listen().onChange(
        function (p) {
            play_pointer = p;
            if(play_fps==0){
                force_move_all(play_pointer)
            }
            console.log('p changed')
    });
    BarFolder.open();

    window.addEventListener('resize', onWindowResize);
}



function parse_style(str){
    //E.g. >>flash('lightning',src=0.00000000e+00,dst=1.00000000e+01,dur=1.00000000e+00)
    let re_style = />>set_style\('(.*)'/
    let re_res = str.match(re_style)
    let style = re_res[1]
    if (style=="grid"){
        scene.children.filter(function (x){return (x.type == 'GridHelper')}).forEach(function(x){
            x.visible = true
        })
    }else if(style=="nogrid"){
        scene.children.filter(function (x){return (x.type == 'GridHelper')}).forEach(function(x){
            x.visible = false
        })
    }else if(style=="gray"){
        scene.background = new THREE.Color(0xa0a0a0);
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
        scene.add( particles );
    }else if(style=='earth'){
        var onRenderFcts= [];
        var light	= new THREE.AmbientLight( 0x222222 )
        scene.add( light )
    
        var light	= new THREE.DirectionalLight( 0xffffff, 1 )
        light.position.set(5,5,5)
        scene.add( light )
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
        scene.add(containerEarth)
    
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
        // new THREEx.addAtmosphereMaterial2DatGui(material, datGUI)
    
        // var earthCloud	= THREEx.Planets.createEarthCloud()
        // earthCloud.receiveShadow	= true
        // earthCloud.castShadow	= true
        // containerEarth.add(earthCloud)
        // onRenderFcts.push(function(delta, now){
        //     earthCloud.rotation.y += 1/8 * delta;		
        // })
        renderer.shadowMapEnabled	= true
    }
}

init();
animate();