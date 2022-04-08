
var init_cam_f1 = false;
var init_cam_f2 = false;


function getTextBackground(text_object, parsed_obj_info){
    text_object.geometry.computeBoundingBox();
    let dx = text_object.geometry.boundingBox.max.x - text_object.geometry.boundingBox.min.x;
    let dy = text_object.geometry.boundingBox.max.y - text_object.geometry.boundingBox.min.y;
    
    if (isFinite(dx)){
        const geometry = new THREE.PlaneGeometry( dx*1.15, dy*1.15 );
        // const yMid = - 0.5 * dy;
        // geometry.translate( 0, yMid, 0 );
        const material = new THREE.MeshBasicMaterial( {color: parsed_obj_info['label_bgcolor'], side: THREE.DoubleSide, transparent: true, opacity:0.5} );
        const plane = new THREE.Mesh( geometry, material );
        plane.renderOrder = 127;
        return plane;
    }else{
        return null;
    }

}

function detach_dispose(remove_what, from_where){
    from_where.remove(remove_what);
    mesh_dispose(remove_what)
    return
}


function mesh_dispose(mesh){
    if (mesh.material.constructor === Array){ for (i=0;i<mesh.material.length;i++){
        mesh.material[i].dispose();}
    }else{
        mesh.material.dispose();
    }
    mesh.geometry.dispose();
}

// assign or replace
function AoR(obj, new_obj){
    if(!obj){
        return new_obj;
    }
    else{
        mesh_dispose(obj)
        obj.material = new_obj.material;
        obj.geometry = new_obj.geometry;
        new_obj = null;
        return obj;
    }
}

function text_object_centering(text_object) {
    text_object.geometry.computeBoundingBox();
    const xMid = -0.5 * (text_object.geometry.boundingBox.max.x + text_object.geometry.boundingBox.min.x);
    const yMid = -0.5 * (text_object.geometry.boundingBox.max.y + text_object.geometry.boundingBox.min.y);
    let dx = text_object.geometry.boundingBox.max.x - text_object.geometry.boundingBox.min.x;
    text_object.geometry.translate(xMid, yMid, dx / 50);
}
function get_text_geometry(object, text, parsed_obj_info){
    let text_size = parsed_obj_info['label_size'];
    let font_size = (text_size==null)?object.generalSize/1.5:text_size/1.5
    return new THREE.ShapeGeometry(window.glb.font.generateShapes( text, font_size ));
}


function get_text_material(object, text, parsed_obj_info){
    let label_opacity = parsed_obj_info['label_opacity'];
    let textcolor = parsed_obj_info['label_color'];
    return new THREE.MeshBasicMaterial( {
        color: textcolor,
        transparent: (label_opacity!=1),
        opacity: label_opacity,
        side: THREE.DoubleSide
    });
}
function get_text_material_arr(object, text, parsed_obj_info, color_arr){
    let label_opacity = parsed_obj_info['label_opacity'];

    let res = []
    for (i=0;i<color_arr.length;i++){
        res.push(new THREE.MeshBasicMaterial( {
            color: color_arr[i],
            transparent: (label_opacity!=1),
            opacity: label_opacity,
            side: THREE.DoubleSide
        }))
    }

    return res
}
function makeClearText_old(object, text, parsed_obj_info){
    if(object.text_object){init=true;}
    else{init=false;}

    object.text_object = AoR(
        object.text_object,
        new THREE.Mesh( 
            get_text_geometry(object, text, parsed_obj_info), 
            get_text_material(object, text, parsed_obj_info) 
        )
    );

    text_object_centering(object.text_object)

    if (!init){

        // (init) fixed id and offset
        object.text_object.my_id = '_text_' + object.my_id
        if (!object.label_offset){object.text_object.position.set(object.generalSize, object.generalSize, -object.generalSize);}
        else{object.text_object.position.set(object.label_offset[0], object.label_offset[2], -object.label_offset[1]);}
        // (init) renderOrder
        object.text_object.renderOrder = 128
    }
    if (parsed_obj_info['label_bgcolor']){
        object.text_object.background = AoR(
            object.text_object.background,
            getTextBackground(object.text_object, parsed_obj_info)
        );
    }

    if (!init){
        // (init) add objects
        object.add(object.text_object)
        if (object.text_object.background){object.text_object.add(object.text_object.background)}
    }

}

function my_match_all(text, pattern){
    const pattern_color = get_reg_exp_g(pattern);
    let match_color_ = text.matchAll(pattern_color);
    let match_color = [];
    while(true){
        let x = match_color_.next();
        if (x.done){break;}
        match_color.push(x.value[1])
    }
    return match_color;
}



function get_text_geometry_arr(object, text, parsed_obj_info, each_color, word_color, match_color){
    let text_size = parsed_obj_info['label_size'];
    let font_size = (text_size==null)?object.generalSize/1.5:text_size/1.5;
    let shape_arr = window.glb.font.myGenerateShapes( text, font_size, each_color, word_color );
    // let geo_res = [];
    let nShapeEachColor = []
    let all_shape = []
    for(i=0; i<match_color.length; i++){
        let color = match_color[i]
        nShapeEachColor.push(shape_arr[color].length)
        Array.prototype.push.apply( all_shape ,  shape_arr[color] );
    }
    let geo = new THREE.ShapeGeometry(all_shape);
    
    let sumpren = 0;
    for(c=0; c<nShapeEachColor.length; c++){
        let n = nShapeEachColor[c];
        for(j=sumpren;j<sumpren+n;j++){
            geo.groups[j].materialIndex = c;
        }
        sumpren += n;
    }

    return geo;
}



function makeClearText(object, text, parsed_obj_info){
    if(object.text_object){init=true;}
    else{init=false;}

    let match_color = my_match_all(text, '<(.*?)>');
    let match_text = my_match_all(text, '>([\\s\\S]*?)<');

    if (match_color.length==0 || match_color.length != match_text.length+1){
        makeClearText_old(object, text, parsed_obj_info);
        return;
    }
    match_color = match_color.slice(0, match_color.length-1)

    each_color_list = []
    each_color = {} // 颜色字典
    word_color = [] // 各个字的颜色
    text_raw = '' // 去除颜色标识的文字

    for(i=0; i < match_color.length; i++){
        let color_ = match_color[i];
        let text_ = match_text[i];
        each_color[color_] = []
        for(j=0; j < text_.length;j++){
            word_color.push(color_)
        }
        let color_already_in_list = (each_color_list.indexOf(color_) >= 0)
        if (!color_already_in_list){
            each_color_list.push(color_);
        }
        text_raw = text_raw + text_
    }

    let material = get_text_material_arr(object, text_raw, parsed_obj_info, each_color_list);
    let geometry = get_text_geometry_arr(object, text_raw, parsed_obj_info, each_color, word_color, each_color_list);

    object.text_object = AoR(
        object.text_object,
        new THREE.Mesh( 
            geometry,
            material
        )
    );

    text_object_centering(object.text_object)

    if (!init){

        // (init) fixed id and offset
        object.text_object.my_id = '_text_' + object.my_id
        if (!object.label_offset){object.text_object.position.set(object.generalSize, object.generalSize, -object.generalSize);}
        else{object.text_object.position.set(object.label_offset[0], object.label_offset[2], -object.label_offset[1]);}
        // (init) renderOrder
        object.text_object.renderOrder = 128
    }
    if (parsed_obj_info['label_bgcolor']){
        object.text_object.background = AoR(
            object.text_object.background,
            getTextBackground(object.text_object, parsed_obj_info)
        );
    }

    if (!init){
        // (init) add objects
        object.add(object.text_object)
        if (object.text_object.background){object.text_object.add(object.text_object.background)}
    }
 

}


function purge_core_obj(i){
    let core_obj = window.glb.core_Obj[i]
    // 清除text
    if (core_obj.text_object){
        if (core_obj.text_object.background){
            detach_dispose(core_obj.text_object.background, from_where=core_obj.text_object)
            core_obj.text_object.background = null
        }
        detach_dispose(core_obj.text_object, from_where=core_obj);
        core_obj.text_object = null;
    }
    // 清除历史轨迹
    if (core_obj.track_init){
        detach_dispose(core_obj.track_his.mesh, from_where=window.glb.scene);
        core_obj.track_his.mesh = null;
        core_obj.track_his = null;
        core_obj.track_init = false;
    }

    detach_dispose(core_obj, from_where=window.glb.scene);
    core_obj = null
    window.glb.core_Obj[i] = null;

    window.glb.core_Obj.splice(i,1); // remove ith object
}

//修改颜色
function changeCoreObjColor(object, color_str){
    const colorjs = color_str;
    object.material.color.set(colorjs)
    object.color_str = color_str;
}
//修改大小
function changeCoreObjSize(object, size){
    let ratio_ = (size/object.initialSize)
    object.scale.x = ratio_
    object.scale.y = ratio_
    object.scale.z = ratio_
    object.currentSize = size
}
//添加形状句柄
MAX_HIS_LEN = 5120
function addCoreObj(my_id, color_str, geometry, material, x, y, z, ro_x, ro_y, ro_z, currentSize, label_marking, label_color, opacity, track_n_frame, parsed_obj_info=null){
    const object = new THREE.Mesh(geometry, material);
    object.my_id = my_id;
    object.color_str = color_str;
    object.position.x = x; 
    object.position.y = y; 
    object.position.z = z; 
    object.next_pos = new THREE.Vector3(0.0,0.0,0.0); object.next_pos.copy(object.position);
    object.prev_pos = new THREE.Vector3(0.0,0.0,0.0); object.prev_pos.copy(object.position);

    object.rotation.x = ro_x;  
    object.rotation.y = ro_y;  
    object.rotation.z = ro_z;  
    object.rotation.reorder(parsed_obj_info['ro_order']);

    object.next_ro = {x:ro_x, y:ro_y, z:ro_z};
    object.prev_ro = {x:ro_x, y:ro_y, z:ro_z};

    object.initialSize = 1
    object.currentSize = currentSize
    object.generalSize = 1
    object.prev_size = currentSize
    object.next_size = currentSize

    object.scale.x = 1; 
    object.scale.y = 1; 
    object.scale.z = 1; 
    object.label_marking = label_marking
    object.label_color = label_color

    object.next_opacity = opacity
    object.prev_opacity = opacity
    object.material.transparent = (opacity==1)?false:true
    object.material.opacity = opacity
    object.renderOrder = parsed_obj_info['renderOrder']
    object.renderOrder = (opacity==0)?256:object.renderOrder
    object.track_n_frame = track_n_frame
    object.track_init = false
    object.track_tension = parsed_obj_info['track_tension']
    object.track_color = parsed_obj_info['track_color']
    // 即刻应用label_offset
    object.label_offset = parsed_obj_info['label_offset']
    object.fade_step = parsed_obj_info['fade_step']

    if (!init_cam_f1){
        init_cam_f1=true;
        let h = (currentSize==0)?0.1:currentSize
        window.glb.controls.target.set(object.position.x, -h, object.position.z); // 旋转的焦点在哪0,0,0即原点
        window.glb.controls2.target.set(object.position.x, -h, object.position.z); // 旋转的焦点在哪0,0,0即原点
        window.glb.camera.position.set(object.position.x, h*100, object.position.z)
        window.glb.camera2.position.set(object.position.x, h*100, object.position.z)
    }
    if (label_marking){
        makeClearText(object, object.label_marking, parsed_obj_info)
    }
    // 初始化历史轨迹
    object.his_positions = [];
    for ( let i = 0; i < MAX_HIS_LEN; i ++ ) {
        object.his_positions.push( new THREE.Vector3(x, y, z) );
    }
    window.glb.scene.add(object);
    window.glb.core_Obj.push(object)
}

//选择形状
function choose_geometry(type){
    if (!window.glb.base_geometry[type]){
        alert('The geometry is not defined for name:'+type)
        // console.log('The geometry is not defined for name:'+type+' , or maybe the geometry is still loading!')
        return null
    }else if(window.glb.base_geometry[type]=='loading'){
        console.log('The geometry is not defined for name:' + type + ' , or maybe the geometry is still loading!')
        return null
    }else{
        return window.glb.base_geometry[type]
    }
}

//选择材质
function choose_material(type, color_str){
    if (window.glb.base_material[type]==null){
        return new THREE.MeshLambertMaterial({ color: color_str })
    }
    else{
        return window.glb.base_material[type].clone()
    }
}
function init_cam(){
    console.log('secondary init_cam')
    let x = 0;
    let y = 0;
    let z = 0;
    let size = 0;
    for (let i = 0; i < window.glb.core_Obj.length; i++) {
        x = x+window.glb.core_Obj[i].position.x/window.glb.core_Obj.length
        y = y+window.glb.core_Obj[i].position.y/window.glb.core_Obj.length
        z = z+window.glb.core_Obj[i].position.z/window.glb.core_Obj.length
        size = size + window.glb.core_Obj[i].currentSize/window.glb.core_Obj.length
    }
    let h = (size==0)?0.1:size
    window.glb.controls.target.set(x, -h, z); // 旋转的焦点在哪0,0,0即原点
    window.glb.controls2.target.set(x, -h, z); // 旋转的焦点在哪0,0,0即原点
    window.glb.camera.position.set(x, h*100, z)
    window.glb.camera2.position.set(x, h*100, z)
}

function load_and_compare_prev(object, parsed_obj_info, attribute){
    // this function can only be executed once each update for each attribute!
    if (!object.prev){object.prev={};}
    if (!object.changed){object.changed={};}

    object.changed[attribute] = (object.prev[attribute] != parsed_obj_info[attribute])
    object.prev[attribute] = parsed_obj_info[attribute]
}

//将parsed_obj_info中的位置、旋转、大小、文本、颜色等等变化应用
function apply_update(parsed_obj_info){
    let my_id = parsed_obj_info['my_id']
    
    let name = parsed_obj_info['name']
    let pos_x = parsed_obj_info['pos_x']
    let pos_y = parsed_obj_info['pos_y']
    let pos_z = parsed_obj_info['pos_z']

    let ro_x = parsed_obj_info['ro_x']
    let ro_y = parsed_obj_info['ro_y']
    let ro_z = parsed_obj_info['ro_z']

    let type = parsed_obj_info['type']
    let color_str = parsed_obj_info['color_str']
    let size = parsed_obj_info['size']
    let label_marking = parsed_obj_info['label_marking']
    let label_color = parsed_obj_info['label_color']
    let opacity = parsed_obj_info['opacity']
    let track_n_frame = parsed_obj_info['track_n_frame']
    let track_tension = parsed_obj_info['track_tension']
    let track_color = parsed_obj_info['track_color']


    let object = find_obj_by_id(my_id)
    // 已经创建了对象,setfuture
    if (object) {
        if (!init_cam_f2){
            init_cam_f2 = true;
            init_cam()
        }

        // roll previous
        object.prev_pos.copy(object.next_pos); //  Object.assign({}, object.next_pos); 
        // load next
        object.next_pos.x = pos_x; object.next_pos.y = pos_y; object.next_pos.z = pos_z;

        // roll previous
        object.prev_ro = Object.assign({}, object.next_ro); // roll next
        // load next
        object.next_ro.x = ro_x; object.next_ro.y = ro_y; object.next_ro.z = ro_z;

        // roll previous
        object.prev_size = object.next_size;
        // load next
        object.next_size = size;

        // roll previous
        object.prev_opacity = object.next_opacity;
        // load next
        object.next_opacity = opacity;

        // 即刻应用renderOrder
        object.renderOrder = parsed_obj_info['renderOrder']
        object.renderOrder = (opacity==0)?256:object.renderOrder;

        // 即刻应用label_offset
        object.label_offset = parsed_obj_info['label_offset']



        load_and_compare_prev(object, parsed_obj_info, 'label_marking'); object.label_marking = label_marking
        load_and_compare_prev(object, parsed_obj_info, 'label_color');   object.label_color = label_color
        load_and_compare_prev(object, parsed_obj_info, 'label_opacity'); object.label_opacity = parsed_obj_info['label_opacity']

        // 即刻应用color和text
        if (color_str != object.color_str) {changeCoreObjColor(object, color_str)}
        if (object.changed['label_marking'] || object.changed['label_color'] || object.changed['label_color']) {
            makeClearText(object, object.label_marking, parsed_obj_info);
        }
        // 即刻应用
        object.track_n_frame = track_n_frame
        object.track_color = track_color
        // 即可更新历史轨迹
        object.his_positions.push( new THREE.Vector3(object.prev_pos.x, object.prev_pos.y, object.prev_pos.z) );
        object.his_positions.shift()

    }
    else {
        // create obj
        let currentSize = size;
        let geometry = choose_geometry(type);
        let material = choose_material(type, color_str)
        if (geometry==null){
            console.log('the geometry is still loading!')
            return
        }
        //function (my_id, color_str, geometry, x, y, z, size, label_marking){
        addCoreObj(my_id, color_str, geometry, material,
            pos_x, pos_y, pos_z, ro_x, ro_y, ro_z, 
            currentSize, label_marking, label_color, opacity, track_n_frame, parsed_obj_info)

        // find the object again!
        object = find_obj_by_id(my_id)
    }

    // Apply Instantly, 即刻应用fade_step
    object.fade_step = parsed_obj_info['fade_step'];
    if (object.fade_step){
        object.fade_level = 1.0
    }else{
        object.fade_level = null
    }

}

