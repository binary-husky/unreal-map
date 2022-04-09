function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}
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


    if (window.glb.buf_storage){
        tmp[0] = window.glb.buf_storage + tmp[0]
    }
    window.glb.buf_storage = ''
    if (!buf.endsWith(EOFF)){
        window.glb.buf_storage = tmp.pop()
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
        window.glb.core_L = [];
        clear_everything();
        window.glb.core_Obj = [];
        tmp.splice(0, eofx_i);
    }
    //
    window.glb.core_L = window.glb.core_L.concat(tmp);
    // check memory remaining
    // console.log(parseInt((performance.memory.jsHeapSizeLimit-performance.memory.usedJSHeapSize)/1024/1024),'MB');
    // ; // will give you the JS heap size
    // performance.memory.usedJSHeapSize; // how much you're currently using

    //
    if (window.glb.core_L.length > window.glb.buffer_size){
        window.glb.core_L.splice(0, tmp.length);
        window.glb.play_pointer = window.glb.play_pointer - tmp.length;
        if (window.glb.play_pointer<0){
            window.glb.play_pointer=0;
        }
    }
    window.glb.BarFolder.__controllers[0].max(window.glb.core_L.length);
    return tmp.length;
}
///////////////////////////////// tools  //////////////////////////

// warning: python mod operator is different from js mod operator
function reg_rad(rad){
    let a = (rad + Math.PI) 
    let b = (2 * Math.PI)

    return ((a%b)+b) % b - Math.PI
}

function reg_rad_at(rad, ref){
    return reg_rad(rad-ref) + ref
}


function generateUUID() {
    var d = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = (d + Math.random()*16)%16 | 0;
      d = Math.floor(d/16);
      return (c=='x' ? r : (r&0x3|0x8)).toString(16);
    });
    return uuid;
};