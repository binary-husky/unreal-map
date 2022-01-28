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
window.glb.BarFolder = null;
window.glb.camera = null;
window.glb.stats = null;

window.glb.import_Stats = Stats;
window.glb.import_GUI = GUI;
window.glb.import_OrbitControls = OrbitControls;
window.glb.import_LightningStrike = LightningStrike;
window.glb.renderer = null;
window.glb.controls=null;
// var window.glb.scene;
// 历史轨迹
window.glb.core_L = [];
window.glb.parsed_core_L = []
window.glb.core_Obj = [];
window.glb.flash_Obj = [];
window.glb.base_geometry = {};
// global vars
window.glb.play_fps = 5;
window.glb.play_pointer = 0; // 位置读取指针
window.glb.solid_pointer = 0; // 过渡动画前置位置的指针
window.glb.sp_future = 0;   // 用于辅助确定 solid_pointer
window.glb.dt_threshold = 1 / window.glb.play_fps;

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
        'reset to read new': null,
        'pause': null,
        'next frame': null,
        'previous frame': null,
        'ppt step': null,
        'loop to start': false
};




//////////////////////main read function//////////////////////////
var coreReadFunc = function (auto_next=true) {
    if (transfer_ongoing){
        // console.log('ongoing')
        req_interval = req_interval+1;
        window.glb.panelSettings['data req interval'] = req_interval
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
        // console.log('loaded')
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
            window.glb.panelSettings['data req interval'] = req_interval
        }else{
            req_interval = 0
            window.glb.panelSettings['data req interval'] = req_interval
        }

        transfer_ongoing = false;
    };
    request.send();
    // console.log('send req')
    transfer_ongoing = true;
    if(auto_next && !DEBUG){setTimeout(coreReadFunc, req_interval*1000);}
    // console.log('next update '+req_interval)
}
window.glb.panelSettings['reset to read new'] = function (){
    if (transfer_ongoing){
        request.abort()
        transfer_ongoing = false;
    }
    coreReadFunc(false)
}
setTimeout(coreReadFunc, 100);

function change_fps(fps) {
    window.glb.play_fps = fps;
    dt_since = 0;
    window.glb.dt_threshold = 1 / window.glb.play_fps;
}

function pause_play(){
    if(window.glb.panelSettings['play fps']>0){
        window.glb.prev_fps = window.glb.panelSettings['play fps'];
    }
    window.glb.play_pointer = window.glb.solid_pointer;
    force_move_all(window.glb.solid_pointer)
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


function removeEntity(object) {
    var selectedObject = window.glb.scene.getObjectByName(object.name);
    window.glb.scene.remove(selectedObject);
}







function parse_time_step(pp){
    if(window.glb.parsed_core_L[pp]) {
        buf_str = window.glb.core_L[pp]
        parse_update_env(buf_str)
        parse_update_without_re(pp)
        parse_update_flash(buf_str)
    }else{
        buf_str = window.glb.core_L[pp]
        parse_init(buf_str)
        parse_update_env(buf_str)
        parse_update_core(buf_str, pp)
        parse_update_flash(buf_str)
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




function change_position_rotation_size(object, percent, override){
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

    if(override){
        object.prev_pos.x = object.next_pos.x; object.prev_pos.y = object.next_pos.y; object.prev_pos.z = object.next_pos.z
        object.prev_ro.x = object.next_ro.x; object.prev_ro.y = object.next_ro.y; object.prev_ro.z = object.next_ro.z
        object.prev_opacity = object.next_opacity; object.prev_size = object.next_size;
    }
}

// called according to fps
function set_next_play_frame() {
    if (window.glb.core_L.length == 0) { return; }
    if (window.glb.play_pointer >= window.glb.core_L.length) {
        window.glb.play_pointer = window.glb.panelSettings['loop to start']?0:window.glb.core_L.length-1;
    }
    parse_time_step(window.glb.play_pointer)
    window.glb.solid_pointer = window.glb.sp_future; window.glb.panelSettings['play pointer'] = window.glb.solid_pointer; window.glb.sp_future = window.glb.play_pointer;
    console.log('set_next_play_frame sp:'+window.glb.solid_pointer+' spf:'+window.glb.sp_future)

    window.glb.play_pointer = window.glb.play_pointer + 1
    if (window.glb.play_pointer >= window.glb.core_L.length) {
        window.glb.play_pointer = window.glb.panelSettings['loop to start']?0:window.glb.core_L.length-1;
    }
}
function force_move_all(pp){ // 手动调整进度条时触发
    parse_time_step(pp)
    window.glb.solid_pointer = pp; window.glb.panelSettings['play pointer'] = window.glb.solid_pointer; window.glb.sp_future = pp;
    console.log('set_next_play_frame sp:'+window.glb.solid_pointer+' spf:'+window.glb.sp_future)
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        object.prev_pos.x = object.next_pos.x; object.prev_pos.y = object.next_pos.y; object.prev_pos.z = object.next_pos.z
        object.prev_ro.x = object.next_ro.x; object.prev_ro.y = object.next_ro.y; object.prev_ro.z = object.next_ro.z
        object.prev_opacity = object.next_opacity; object.prev_size = object.next_size;
        change_position_rotation_size(object, 1, true)
    }	
}
window.glb.force_move_all = force_move_all

function move_to_future(percent, override=false) {
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        let object = window.glb.core_Obj[i]
        change_position_rotation_size(object, percent, override)
    }
}





function animate() {
    requestAnimationFrame(animate);
    render();
    window.glb.stats.update();
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
    check_flash_life_cyc(delta)
    window.glb.renderer.render(window.glb.scene, window.glb.camera);
}





init();
animate();