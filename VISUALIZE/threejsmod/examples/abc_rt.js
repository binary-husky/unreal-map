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
import { LineMaterial } from './jsm/lines/LineMaterial.js';
import { LineGeometry } from './jsm/lines/LineGeometry.js';
import { FBXLoader } from './jsm/loaders/FBXLoader.js';
import { Font, FontLoader } from './jsm/loaders/FontLoader.js';
import { TTFLoader } from './jsm/loaders/TTFLoader.js';
import { Line2 } from './jsm/lines/Line2.js';
window.glb = Object()
window.glb.clock = new THREE.Clock();
window.glb.scene = null;
window.glb.buf_storage = ''
window.glb.BarFolder = null;
window.glb.camera = null;
window.glb.camera2 = null;
window.glb.stats = null;
window.glb.font = null;

window.glb.import_Stats = Stats;
window.glb.import_GUI = GUI;
window.glb.import_OrbitControls = OrbitControls;
window.glb.import_LightningStrike = LightningStrike;
window.glb.import_LineMaterial = LineMaterial;
window.glb.import_LineGeometry = LineGeometry;
window.glb.import_Line2 = Line2;
window.glb.import_Font = Font;
window.glb.import_FontLoader = FontLoader;
window.glb.import_TTFLoader = TTFLoader;
window.glb.import_FBXLoader = FBXLoader;
window.glb.renderer = null;
window.glb.controls=null;
window.glb.controls2=null;
// var window.glb.scene;
window.glb.core_L = [];
window.glb.core_Obj = [];
window.glb.line_Obj = [];
// window.glb.text_Obj = [];
window.glb.flash_Obj = [];
window.glb.base_geometry = {};
window.glb.base_material = {};
// global vars
window.glb.play_fps = 144;
window.glb.play_pointer = 0; // 位置读取指针
window.glb.solid_pointer = 0; // 过渡动画前置位置的指针
window.glb.sp_future = 0;   // 用于辅助确定 solid_pointer
window.glb.dt_threshold = 1 / window.glb.play_fps;
var client_uuid = generateUUID();

//1e4
window.glb.buffer_size = 1e4

var dt_since = 0;
var buf_str = '';
var DEBUG = false;
var transfer_ongoing = false;
var ppt_mode = 0;
// GUI
var request=null;
var req_interval=2.0;
window.glb.panelSettings = {
    'play fps': window.glb.play_fps,
    'play pointer':0,
    'data req interval': req_interval,
    'auto fps': false,
    // 'reset to read new': null,
    'pause': null,
    'next frame': null,
    'previous frame': null,
    'freeze': null,
    'ppt step': null,
    'loop to start': false,
    'smooth control': true,
    'show camera orbit': false,
    'use orthcam': false
};


var pre_rec_time = -1
var incoming_per_sec = -1
var recent_incoming_steam = []

function auto_fps_experimental(recent_incoming_steam){
    if (window.glb.panelSettings['auto fps']){
        let N=0; let dt=0;
        for (let i=0; i<recent_incoming_steam.length; i++){
            dt += recent_incoming_steam[i]['time'];
            N += recent_incoming_steam[i]['n_data'];
        }
        let incoming_per_sec = N/dt


        // automatically change fps
        let need_fps_change = false;
        let nframe_to_play = window.glb.core_L.length - window.glb.play_pointer;

        let max_speed_nframe = 1250;
        let max_speed = incoming_per_sec*5;

        let min_speed_nframe = 250;
        let min_speed = incoming_per_sec*0.5;
        let control_fps = min_speed + (nframe_to_play - min_speed_nframe) * (max_speed-min_speed) / (max_speed_nframe-min_speed_nframe)
        if (nframe_to_play<min_speed_nframe){
            control_fps = min_speed;
        }
        console.log('pre_rec_time:'+incoming_per_sec+'\tnframe_to_play:'+nframe_to_play+'\tcontrol_fps:'+control_fps)

        if(Math.abs(window.glb.panelSettings['play fps'] - control_fps)>=1){
            need_fps_change = true;
            window.glb.panelSettings['play fps'] = control_fps;
        }
        // if ((window.glb.core_L.length - window.glb.play_pointer)> window.glb.buffer_size*0.3){ // 20% buffer size
        //     window.glb.panelSettings['play fps'] = incoming_per_sec*5;
        // }else if ((window.glb.core_L.length - window.glb.play_pointer)< window.glb.buffer_size*0.05){ // 20% buffer size
        //     window.glb.panelSettings['play fps'] = incoming_per_sec*0.5;
        // }else if((Math.abs(window.glb.panelSettings['play fps'] - incoming_per_sec)>=1) ){ // 10% ~ 30% buffer size
        //     window.glb.panelSettings['play fps'] = incoming_per_sec;
        // }else{
        //     need_fps_change = false;
        // }

        if (window.glb.panelSettings['play fps']>144){window.glb.panelSettings['play fps']=144;}
        if (window.glb.panelSettings['play fps']<0.1){window.glb.panelSettings['play fps']=0.1;}
        if (need_fps_change) {change_fps(window.glb.panelSettings['play fps']);}
        if (need_fps_change) {console.log('change_fps:'+window.glb.panelSettings['play fps']);}
    }
}

//////////////////////main read function//////////////////////////
// let inflat;
// if (!DEBUG){inflat = pako.inflate(toUint8Arr(request.response),{to: 'string'});}
// else{inflat = request.responseText;}
var gWebSocket = null;
var gWsConn = false;

// let n_new_data = core_update(inflat)
var coreReadFunc = function (auto_next=true) {
    var ws = new WebSocket('ws://localhost:8765');
    ws.binaryType = "arraybuffer";
    // var ws = new WebSocket(window.location.origin.replace('http:', 'ws:').replace('https:', 'wss:') + '/echo');
    ws.onopen = function() {
    };
    ws.onclose = function(evt) {
        alert("Socket connection failed... maybe you can try non-realtime vhmap");
    };
    ws.onmessage = function(event) {
        let inflat;
        inflat = pako.inflate(event.data,{to: 'string'});
        let n_new_data = core_update(inflat);
    };
}


// window.glb.panelSettings['reset to read new'] = function (){
//     if (transfer_ongoing){
//         request.abort()
//         transfer_ongoing = false;
//     }
//     coreReadFunc(false)
// }

setTimeout(coreReadFunc, 100);

function change_fps(fps) {
    if (Math.abs(window.glb.play_fps-fps)<0.5){ return; }
    window.glb.play_fps = fps;
    dt_since = 0;
    window.glb.dt_threshold = 1 / window.glb.play_fps;
}

function pause_play(test_freeze=false){
    if(window.glb.panelSettings['play fps']>0){
        window.glb.prev_fps = window.glb.panelSettings['play fps'];
    }
    window.glb.play_pointer = window.glb.solid_pointer;
    force_move_all(window.glb.solid_pointer, test_freeze)
    window.glb.panelSettings['play fps'] = 0; 
    change_fps(0);
}

function resume_play(){
    start_transition()
    window.glb.panelSettings['play fps'] = window.glb.prev_fps; 
    change_fps(window.glb.prev_fps);
}

window.glb.panelSettings['pause'] = function (){
    if (window.glb.panelSettings['play fps']>0){
        pause_play()
    }else{
        resume_play()
    }
}
window.glb.panelSettings['freeze'] = function (){
    if (window.glb.panelSettings['play fps']>0){
        pause_play(true)
    }else{
        resume_play()
    }
}
window.glb.panelSettings['next frame'] = function (){
    // pause play
    if(window.glb.play_fps!=0){
        pause_play()
    } 
    // fix window.glb.play_pointer
    if (window.glb.solid_pointer < window.glb.core_L.length-1)
    {
        window.glb.play_pointer = window.glb.solid_pointer + 1;
    }
    // sync ui
    window.glb.panelSettings['play pointer'] = window.glb.solid_pointer + 1;
    // move all obj
    force_move_all(window.glb.play_pointer)
}

window.glb.panelSettings['previous frame'] = function (){
    // pause play
    if(window.glb.play_fps!=0){
        pause_play()
    } 
    if(window.glb.solid_pointer<=0) {return}
    // fix window.glb.play_pointer
    window.glb.play_pointer = window.glb.solid_pointer - 1;
    // sync ui
    window.glb.panelSettings['play pointer'] = window.glb.solid_pointer - 1;
    // move all obj
    force_move_all(window.glb.play_pointer)
}

window.glb.panelSettings['ppt step'] = function (){
    // pause play
    if(window.glb.play_fps!=0){
        pause_play()
    } 
    // 
    ppt_mode = 2
    let ppt_fps = 1/3
    window.glb.panelSettings['play fps'] = ppt_fps; 
    dt_since = 1/ppt_fps - 0.0001;
    window.glb.play_fps = ppt_fps;
    window.glb.dt_threshold = 1 / ppt_fps;
    set_next_play_frame(); // parse t, finish t --> t
    move_to_future(1);
    // wait parse t+1: t --> t+1
}


////////////////////////////////////////////////////////////







function parse_time_step(pp){
    buf_str = window.glb.core_L[pp];
    parse_init(buf_str);
    parse_update_env(buf_str);
    parse_update_core(buf_str, pp);
    parse_update_flash(buf_str);
}





let currentTime = 0;
function check_flash_life_cyc(delta_time){
    // delta_time: seconds
    currentTime += delta_time
    for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
        let nowTime = new Date();
        let dTime = nowTime - window.glb.flash_Obj[i]['create_time']   // 相差的毫秒数
        if (dTime>=window.glb.flash_Obj[i]['dur']*1000 && window.glb.flash_Obj[i]['valid']){
            detach_dispose(window.glb.flash_Obj[i]['mesh'], window.glb.scene);
            window.glb.flash_Obj[i]['mesh'] = null;
            window.glb.flash_Obj[i]['valid'] = false;
        }
    }
    window.glb.flash_Obj = window.glb.flash_Obj.filter(function (s) {return s['valid'];});
    for (let i = window.glb.flash_Obj.length-1; i>=0; i--) {
        window.glb.flash_Obj[i]['update_target'].update(currentTime)
    }
}



// force_move_all(pp, test_freeze=false) --> change_position_rotation_size(object, 1, true, true)
// move_to_future(percent, override=false)  --> change_position_rotation_size(object, percent, override, false)
function change_position_rotation_size(object, percent, override, reset_track=false){
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
    if (object.material.opacity!=opacity){ 
        object.material.opacity = opacity;
        object.material.transparent = (opacity==1)?false:true;
        object.renderOrder = (opacity==0)?256:object.renderOrder;
    }
    changeCoreObjSize(object, size)

    if(override){
        object.prev_pos.x = object.next_pos.x; object.prev_pos.y = object.next_pos.y; object.prev_pos.z = object.next_pos.z
        object.prev_ro.x = object.next_ro.x; object.prev_ro.y = object.next_ro.y; object.prev_ro.z = object.next_ro.z
        object.prev_opacity = object.next_opacity; object.prev_size = object.next_size;

    }
    if(reset_track){
        // 初始化历史轨迹
        object.his_positions = [];
        for ( let i = 0; i < MAX_HIS_LEN; i ++ ) {
            object.his_positions.push( new THREE.Vector3(object.prev_pos.x, object.prev_pos.y, object.prev_pos.z) );
        }
    }
    plot_obj_track(object)
}

// plot_obj_track
function plot_obj_track(object){
    if (object.track_n_frame == 0) {return;}
    if (object.track_init){
        // 更新轨迹
        let curve = object.track_his
        const positions = [];
        for ( let i = (object.his_positions.length - object.track_n_frame); i < object.his_positions.length; i++) {
            positions.push(object.his_positions[i]);
        }
        positions.push(object.position)
        curve.points = positions
        curve.tension = object.track_tension;
        const position = curve.mesh.geometry.attributes.position;
        const point = new THREE.Vector3();
        if(object.track_n_frame>2560){alert("track_n_frame is too large, must < 2560")}
        for ( let i = 0; i < curve.arc_seg; i ++ ) {
            const t = i / ( curve.arc_seg - 1 );
            curve.getPoint( t, point );
            position.setXYZ( i, point.x, point.y, point.z );
        }
        if(curve.current_color!=object.track_color){
            curve.current_color=object.track_color
            changeCoreObjColor(curve.mesh, object.track_color)
        }
        curve.mesh.geometry.computeBoundingSphere();
        position.needsUpdate = true;
    }else{
        // 初始化轨迹  
        object.track_init = true;
        const positions = [];
        for ( let i = (object.his_positions.length - object.track_n_frame); i < object.his_positions.length; i++) {
            positions.push(object.his_positions[i]);
        }
        positions.push(object.position)
        object.track_his = new THREE.CatmullRomCurve3( positions );
        object.track_his.curveType = 'catmullrom';
        // load positions_catmull
        let curve = object.track_his
        curve.arc_seg = (object.track_n_frame > 250)? 5120:512
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( curve.arc_seg * 3 ), 3 ) );
        curve.mesh = new THREE.Line( geometry.clone(), new THREE.LineBasicMaterial( {
            color: object.track_color,
        } ) );
        curve.current_color = object.track_color
        curve.tension = object.track_tension;
        const position = curve.mesh.geometry.attributes.position;
        const point = new THREE.Vector3();
        for ( let i = 0; i < curve.arc_seg; i ++ ) {
            const t = i / ( curve.arc_seg - 1 );
            curve.getPoint( t, point );
            position.setXYZ( i, point.x, point.y, point.z );
        }
        curve.mesh.geometry.computeBoundingSphere();
        window.glb.scene.add(curve.mesh);
    }

}

// called according to fps
function set_next_play_frame() {
    if (window.glb.core_L.length == 0) { return; }
    if (window.glb.play_pointer >= window.glb.core_L.length) {
        window.glb.play_pointer = window.glb.panelSettings['loop to start']?0:window.glb.core_L.length-1;
        if(!window.glb.panelSettings['loop to start'] && window.glb.play_pointer==(window.glb.core_L.length-1)){ return; }
    }
    parse_time_step(window.glb.play_pointer)
    window.glb.solid_pointer = window.glb.sp_future; window.glb.panelSettings['play pointer'] = window.glb.solid_pointer; window.glb.sp_future = window.glb.play_pointer;
    // console.log('set_next_play_frame sp:'+window.glb.solid_pointer+' spf:'+window.glb.sp_future)

    window.glb.play_pointer = window.glb.play_pointer + 1
    if (window.glb.play_pointer >= window.glb.core_L.length) {
        window.glb.play_pointer = window.glb.panelSettings['loop to start']?0:window.glb.core_L.length-1;
    }
}

function force_move_all(pp, test_freeze=false){ // 手动调整进度条时触发
    parse_time_step(pp)
    window.glb.solid_pointer = pp; window.glb.panelSettings['play pointer'] = window.glb.solid_pointer; window.glb.sp_future = pp;
    console.log('set_next_play_frame sp:'+window.glb.solid_pointer+' spf:'+window.glb.sp_future)
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        if(!test_freeze){
            object.prev_pos.x = object.next_pos.x; object.prev_pos.y = object.next_pos.y; object.prev_pos.z = object.next_pos.z
            object.prev_ro.x = object.next_ro.x; object.prev_ro.y = object.next_ro.y; object.prev_ro.z = object.next_ro.z
            object.prev_opacity = object.next_opacity; object.prev_size = object.next_size;
        }else{
            object.prev_pos = object.position.clone()
            object.next_pos = object.position.clone()
            object.prev_ro.x = object.rotation.x; object.prev_ro.y = object.rotation.y; object.prev_ro.z = object.rotation.z
            object.next_ro.x = object.rotation.x; object.next_ro.y = object.rotation.y; object.next_ro.z = object.rotation.z
            object.prev_opacity = object.material.opacity
            object.next_opacity = object.material.opacity
            object.prev_size = object.currentSize
            object.next_size = object.currentSize
        }
        let reset_track = !test_freeze
        change_position_rotation_size(object, 1, true, reset_track)
    }	
}
window.glb.force_move_all = force_move_all

function move_to_future(percent, override=false) {
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        change_position_rotation_size(object, percent, override, false)
    }
}





function animate() {
    requestAnimationFrame(animate);
    render();
    window.glb.stats.update();
    window.glb.controls.update()
}

function start_transition(){
    move_to_future(1.0, true);
    dt_since = 0;
    set_next_play_frame();
    move_to_future(0.0);
    if(ppt_mode>0){
        ppt_mode = ppt_mode - 1
        if(ppt_mode<=0){pause_play()}
        // window.glb.panelSettings['play fps'] = 0; 
        // change_fps(0)
    }

}

function render() {
    const delta = window.glb.clock.getDelta();
    dt_since = dt_since + delta;
    let percent = Math.min(dt_since / window.glb.dt_threshold, 1.0);
    // if(ppt_mode>0){
    //     console.log(percent, dt_since, window.glb.dt_threshold)
    // }
    if (dt_since > window.glb.dt_threshold) {
        start_transition()
    }
    else {
        let percent = Math.min(dt_since / window.glb.dt_threshold, 1.0);
        move_to_future(percent);
    }
    check_flash_life_cyc(delta) // consider moving it somewhere else?

    sprite_like_txt_update()

    if(window.glb.panelSettings['use orthcam']) {window.glb.renderer.render(window.glb.scene, window.glb.camera2);}
    else{
        show_cam_orbit();
        window.glb.renderer.render(window.glb.scene, window.glb.camera);
    }

}

function sprite_like_txt_update(){
    var lookAtVector = new THREE.Vector3(0, 0, -1);
    lookAtVector.applyQuaternion(window.glb.camera.quaternion);
    for (let i = window.glb.core_Obj.length-1; i>=0 ; i--) {
        if (!window.glb.core_Obj[i].text_object){continue;}
        let text_object = window.glb.core_Obj[i].text_object
        let target = new THREE.Vector3(); 
        text_object.getWorldPosition(target)
        text_object.lookAt(target.add(lookAtVector.clone().negate()));
        let object = text_object.parent;
        let new_rel_pos;
        if (!object.label_offset){
            new_rel_pos = new THREE.Vector3(object.generalSize, object.generalSize, -object.generalSize);
        }else{
            new_rel_pos = new THREE.Vector3(object.label_offset[0], object.label_offset[2], -object.label_offset[1]);
        }
        let q = object.quaternion.clone().invert()
        new_rel_pos.applyQuaternion(q)
        text_object.position.set(new_rel_pos.x,new_rel_pos.y,new_rel_pos.z)
    }
}

var cam_orbit_deleted = true;
function show_cam_orbit(){
    if(window.glb.panelSettings['show camera orbit']) {
        let target = window.glb.controls.target
        let distance = window.glb.controls.getDistance()
        apply_simple_line_update(find_lineobj_by_id('cam_x'), {
            'x_arr': [target.x,target.x+distance*0.2],
            'y_arr': [target.y,target.y],
            'z_arr': [target.z,target.z],
            'my_id': 'cam_x', 'color_str': 'Red',
        });
        apply_simple_line_update(find_lineobj_by_id('cam_y'),{
            'x_arr': [target.x,target.x],
            'y_arr': [target.y,target.y+distance*0.2],
            'z_arr': [target.z,target.z],
            'my_id': 'cam_y', 'color_str': 'Blue',
        });
        apply_simple_line_update(find_lineobj_by_id('cam_z'),{
            'x_arr': [target.x,target.x],
            'y_arr': [target.y,target.y],
            'z_arr': [target.z-distance*0.2,target.z],
            'my_id': 'cam_z', 'color_str': 'Green',
        });
        cam_orbit_deleted = false
    }else if(!cam_orbit_deleted){
        let cam_x = find_lineobj_by_id('cam_x'); if (cam_x){window.glb.scene.remove(cam_x.mesh); window.glb.line_Obj.remove(cam_x);}
        let cam_y = find_lineobj_by_id('cam_y'); if (cam_y){window.glb.scene.remove(cam_y.mesh); window.glb.line_Obj.remove(cam_y);}
        let cam_z = find_lineobj_by_id('cam_z'); if (cam_z){window.glb.scene.remove(cam_z.mesh); window.glb.line_Obj.remove(cam_z);}
        cam_orbit_deleted=true;
    }
}


init();
animate();