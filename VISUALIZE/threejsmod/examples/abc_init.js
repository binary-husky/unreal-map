function onWindowResize() {
    window.glb.camera.aspect = window.innerWidth / window.innerHeight;
    window.glb.camera.updateProjectionMatrix();
    window.glb.renderer.setSize(window.innerWidth, window.innerHeight);
}

function init() {
    window.glb.container = document.createElement('div');
    document.body.appendChild(window.glb.container);
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

    window.glb.renderer = new THREE.WebGLRenderer({ antialias: true });
    window.glb.renderer.setPixelRatio(window.devicePixelRatio);
    window.glb.renderer.setSize(window.innerWidth, window.innerHeight);
    window.glb.container.appendChild(window.glb.renderer.domElement);


    window.glb.stats = new window.glb.import_Stats();
    window.glb.container.appendChild(window.glb.stats.dom);

    window.glb.controls = new window.glb.import_OrbitControls(window.glb.camera, window.glb.renderer.domElement);
    // window.glb.controls.object.up = new THREE.Vector3( 1, 0, 0 )
    window.glb.controls.target.set(0, 0, 0); // 旋转的焦点在哪0,0,0即原点
    window.glb.camera.position.set(0, 50, 0)
    window.glb.controls.update();
    window.glb.controls.autoRotate = false;

    
    const panel = new window.glb.import_GUI( { width: 310 } );
    const Folder1 = panel.addFolder( 'General' );
    // FPS adjust
    Folder1.add(window.glb.panelSettings, 'play fps', 0, 144, 1).listen().onChange(
        function change_fps(fps) {
            window.glb.play_fps = fps;
            dt_since = 0;
            window.glb.dt_threshold = 1 / window.glb.play_fps;
        });
    Folder1.add(window.glb.panelSettings, 'data req interval', 1, 100, 1).listen().onChange(
        function (interval) {
            req_interval = interval;
    });
    Folder1.add( window.glb.panelSettings, 'reset to read new' );
    Folder1.open();

    window.glb.BarFolder = panel.addFolder('Play Pointer');
    window.glb.BarFolder.add(window.glb.panelSettings, 'play pointer', 0, 10000, 1).listen().onChange(
        function (p) {
            window.glb.play_pointer = p;
            if(window.glb.play_fps==0){
                window.glb.force_move_all(window.glb.play_pointer)
            }
    });
    window.glb.BarFolder.add( window.glb.panelSettings, 'pause'          );
    window.glb.BarFolder.add( window.glb.panelSettings, 'next frame'     );
    window.glb.BarFolder.add( window.glb.panelSettings, 'previous frame' );
    window.glb.BarFolder.add( window.glb.panelSettings, 'loop to start' );
    window.glb.BarFolder.add( window.glb.panelSettings, 'ppt step' );
    window.glb.BarFolder.open();


    window.addEventListener('resize', onWindowResize);
}