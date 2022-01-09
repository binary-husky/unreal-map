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
        window.glb.core_L = []
        window.glb.parsed_core_L = []
        for (let i = window.glb.core_Obj.length-1; i>=0; i--) {
            window.glb.scene.remove(window.glb.core_Obj[i]);
        }
        window.glb.core_Obj = []
        tmp.splice(0, eofx_i);
    }
    //
    window.glb.core_L = window.glb.core_L.concat(tmp);
    if (window.glb.core_L.length>1e8){
        window.glb.core_L.splice(0, tmp.length);
        play_pointer = play_pointer - tmp.length;
        if (play_pointer<0){
            play_pointer=0;
        }
    }
    window.glb.BarFolder.__controllers[0].max(window.glb.core_L.length);
    return tmp.length;
}
