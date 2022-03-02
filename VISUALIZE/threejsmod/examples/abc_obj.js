
var init_cam_f1 = false;
var init_cam_f2 = false;
// 设置标签
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

    sprite.renderOrder = 128
    object.add(sprite)
}
function makeClearText_new(object, text, textcolor, text_size=null, label_opacity){
    const matLite = new THREE.MeshBasicMaterial( {
        color: textcolor,
        transparent: (label_opacity!=1),
        // transparent: (parsed_obj_info['opacity']==1)?false:true,
        opacity: label_opacity,
        // opacity: parsed_obj_info['opacity'],
        side: THREE.DoubleSide
    });
    let font_size = (text_size==null)?object.generalSize/1.5:text_size/1.5
    const geometry = new THREE.ShapeGeometry(
        window.glb.font.generateShapes( text, font_size)
    );
    geometry.computeBoundingBox();
    const xMid = - 0.5 * ( geometry.boundingBox.max.x - geometry.boundingBox.min.x );
    geometry.translate( xMid, 0, 0 );
    const text_object = new THREE.Mesh( geometry, matLite );

    text_object.my_id = '_text_' + object.my_id
    window.glb.text_Obj.push(text_object) 

    // sprite.scale.x = 17* object.generalSize
    // sprite.scale.y = 17* object.generalSize
    // sprite.scale.z = 17* object.generalSize
    if (!object.label_offset){
        text_object.position.set(object.generalSize, object.generalSize, -object.generalSize);
    }else{
        text_object.position.set(object.label_offset[0], object.label_offset[2], -object.label_offset[1]);
    }
    text_object.renderOrder = 128


    text_object.update_text = function(object, text, textcolor, text_size=null, label_opacity){
        let font_size = (text_size==null)?object.generalSize/1.5:text_size/1.5
        text_object.geometry = new THREE.ShapeGeometry(
            window.glb.font.generateShapes( text, font_size)
        );
        text_object.material = new THREE.MeshBasicMaterial( {
            color: textcolor,
            transparent: (label_opacity!=1),
            // transparent: (parsed_obj_info['opacity']==1)?false:true,
            opacity: label_opacity,
            // opacity: parsed_obj_info['opacity'],
            side: THREE.DoubleSide
        });
        text_object.geometry.computeBoundingBox();
        const xMid = - 0.5 * ( geometry.boundingBox.max.x - geometry.boundingBox.min.x );
        text_object.geometry.translate( xMid, 0, 0 );

    }
    window.glb.text_Obj.push(text_object)
    object.dynamicTexture2 = text_object
    object.add(text_object)
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

    if (!init_cam_f1){
        init_cam_f1=true;
        let h = (currentSize==0)?0.1:currentSize
        window.glb.controls.target.set(object.position.x, -h, object.position.z); // 旋转的焦点在哪0,0,0即原点
        window.glb.controls2.target.set(object.position.x, -h, object.position.z); // 旋转的焦点在哪0,0,0即原点
        window.glb.camera.position.set(object.position.x, h*100, object.position.z)
        window.glb.camera2.position.set(object.position.x, h*100, object.position.z)
    }
    if (label_marking){
        // makeClearText(object, object.label_marking, object.label_color)
        makeClearText_new(object, object.label_marking, object.label_color, parsed_obj_info['label_size'], parsed_obj_info['label_opacity'])
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
    if (window.glb.base_geometry[type]==null){
        alert('The geometry is not defined for name:'+type+' , or maybe the geometry is still loading!')
        // console.log('maybe the geometry is still loading!')
        return null
    }
    else{
        // console.log('using geo:'+type)
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
    window.glb.controls.target.set(x, -h,    z); // 旋转的焦点在哪0,0,0即原点
    window.glb.controls2.target.set(x, -h,    z); // 旋转的焦点在哪0,0,0即原点
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
function apply_update(object, parsed_obj_info){
    let name = parsed_obj_info['name']
    let pos_x = parsed_obj_info['pos_x']
    let pos_y = parsed_obj_info['pos_y']
    let pos_z = parsed_obj_info['pos_z']

    let ro_x = parsed_obj_info['ro_x']
    let ro_y = parsed_obj_info['ro_y']
    let ro_z = parsed_obj_info['ro_z']

    let type = parsed_obj_info['type']
    let my_id = parsed_obj_info['my_id']
    let color_str = parsed_obj_info['color_str']
    let size = parsed_obj_info['size']
    let label_marking = parsed_obj_info['label_marking']
    let label_color = parsed_obj_info['label_color']
    let opacity = parsed_obj_info['opacity']
    let track_n_frame = parsed_obj_info['track_n_frame']
    let track_tension = parsed_obj_info['track_tension']
    let track_color = parsed_obj_info['track_color']



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
        if (color_str != object.color_str) {
            changeCoreObjColor(object, color_str)
        }
        if (object.changed['label_marking'] || object.changed['label_color'] || object.changed['label_color']) {
            if (!object.dynamicTexture2) {
                // makeClearText(object, object.label_marking, object.label_color)
                makeClearText_new(object, object.label_marking, object.label_color, parsed_obj_info['label_size'], parsed_obj_info['label_opacity'])
            }
            // object.dynamicTexture.clear().drawText(label_marking, 0, +80, object.label_color)
            object.dynamicTexture2.update_text(object, object.label_marking, object.label_color, parsed_obj_info['label_size'], parsed_obj_info['label_opacity'])
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
    }
}


// function apply_text_update(parsed_obj_info){
//     let my_id = '_text_' + parsed_obj_info['my_id'].toString()
//     let text_object = find_text_by_id(my_id)
//     if (!window.glb.font){return;}
//     if (text_object){return;}
//     else{
//         const matLite = new THREE.MeshBasicMaterial( {
//             color: parsed_obj_info['color_str'],
//             transparent: (parsed_obj_info['opacity']==1)?false:true,
//             opacity: parsed_obj_info['opacity'],
//             side: THREE.DoubleSide
//         });

//         const shapes = window.glb.font.generateShapes( parsed_obj_info['message'], parsed_obj_info['size'] );
//         const geometry = new THREE.ShapeGeometry( shapes );
//         geometry.computeBoundingBox();
//         const xMid = - 0.5 * ( geometry.boundingBox.max.x - geometry.boundingBox.min.x );
//         geometry.translate( xMid, 0, 0 );
//         geometry.scale.x = 1;
//         geometry.scale.y = 1;
//         geometry.scale.z = 1;
//         text_object = new THREE.Mesh( geometry, matLite );
//         // text_object.translate(0,0,0);
//         text_object.my_id = my_id
//         window.glb.scene.add(text_object);
//         window.glb.text_Obj.push(text_object)
//     }

// }