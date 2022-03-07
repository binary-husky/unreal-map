function onWindowResize() {
    window.glb.camera.aspect = window.innerWidth / window.innerHeight;
    window.glb.camera2.aspect = window.innerWidth / window.innerHeight;
    window.glb.camera.updateProjectionMatrix();
    window.glb.camera2.updateProjectionMatrix();
    window.glb.renderer.setSize(window.innerWidth, window.innerHeight);
}

function init() {
    window.glb.container = document.createElement('div');
    document.body.appendChild(window.glb.container);
    // 透视相机  Fov, Aspect, Near, Far – 相机视锥体的远平面
    window.glb.camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, 0.001, 10000);
    window.glb.camera2 = new THREE.OrthographicCamera(window.innerWidth/-2,window.innerWidth/2,window.innerHeight/2,window.innerHeight/-2, -1000, 1000 );

    window.glb.scene = new THREE.Scene();

    const grid = new THREE.GridHelper( 500, 500, 0xffffff, 0x555555 );
    grid.position.y = 0; grid.visible = false

    window.glb.scene.add(grid);
    window.glb.light = new THREE.PointLight(0xffffff, 1);
    window.glb.light2 = new THREE.DirectionalLight( 0xffffff, 1 );
    window.glb.scene.add(window.glb.camera);
    window.glb.scene.add(window.glb.camera2);
    window.glb.camera.add(window.glb.light);
    window.glb.camera2.add(window.glb.light2);
    window.glb.camera2.remove(window.glb.light2)

    window.glb.renderer = new THREE.WebGLRenderer({ antialias: true });
    window.glb.renderer.setPixelRatio(window.devicePixelRatio);
    window.glb.renderer.setSize(window.innerWidth, window.innerHeight);
    window.glb.container.appendChild(window.glb.renderer.domElement);

    window.glb.stats = new window.glb.import_Stats();
    window.glb.container.appendChild(window.glb.stats.dom);

    window.glb.controls = new window.glb.import_OrbitControls(window.glb.camera, window.glb.renderer.domElement);
    window.glb.controls2 = new window.glb.import_OrbitControls(window.glb.camera2, window.glb.renderer.domElement);
    window.glb.controls.target.set(0, 0, 0); // 旋转的焦点在哪0,0,0即原点
    window.glb.controls2.target.set(0, 0, 0); // 旋转的焦点在哪0,0,0即原点
    window.glb.camera.position.set(0, 50, 0)
    window.glb.camera2.position.set(0, 50, 0)
    window.glb.controls.update();
    window.glb.controls2.update();
    window.glb.controls.autoRotate = false;
    window.glb.controls2.autoRotate = false;
    window.glb.controls.enabled = true;
    window.glb.controls2.enabled = false;

    window.glb.controls.listenToKeyEvents(window);
    window.glb.controls.enableDamping=true;

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
    // Folder1.add( window.glb.panelSettings, 'reset to read new' );
    Folder1.add( window.glb.panelSettings, 'auto fps' );
    Folder1.open();

    window.glb.BarFolder = panel.addFolder('Play Pointer');
    window.glb.BarFolder.add(window.glb.panelSettings, 'play pointer', 0, 10000, 1).listen().onChange(
        function (p) {
            window.glb.play_pointer = p;
            // if(window.glb.play_fps==0){
            window.glb.force_move_all(window.glb.play_pointer)
            // }
    });
    window.glb.BarFolder.add( window.glb.panelSettings, 'pause'          );
    window.glb.BarFolder.add( window.glb.panelSettings, 'next frame'     );
    window.glb.BarFolder.add( window.glb.panelSettings, 'previous frame' );
    window.glb.BarFolder.add( window.glb.panelSettings, 'loop to start'  );
    window.glb.BarFolder.add( window.glb.panelSettings, 'freeze'          );
    window.glb.BarFolder.add( window.glb.panelSettings, 'ppt step'       );
    window.glb.BarFolder.add( window.glb.panelSettings, 'show camera orbit');
    window.glb.BarFolder.add( window.glb.panelSettings, 'use orthcam' ).listen().onChange(
        function (use_orthcam) {
            if(use_orthcam){
                window.glb.controls.enabled = false ;
                window.glb.controls2.enabled = true ;
                window.glb.camera.remove(window.glb.light);
                window.glb.camera2.add(window.glb.light2);

                window.glb.panelSettings['smooth control'] = false;
                window.glb.controls.enableDamping=false;
                window.glb.controls2.enableDamping=false;
                window.glb.panelSettings['show camera orbit'] = false;
            }
            else{
                window.glb.controls.enabled = true;
                window.glb.controls2.enabled = false;
                window.glb.camera.add(window.glb.light);
                window.glb.camera2.remove(window.glb.light2);
            }
    });
    window.glb.BarFolder.add( window.glb.panelSettings, 'smooth control' ).listen().onChange(
        function (use_damp) {
            if(use_damp){
                window.glb.controls.enableDamping=true;
                window.glb.controls2.enableDamping=true;
            }
            else{
                window.glb.controls.enableDamping=false;
                window.glb.controls2.enableDamping=false;
            }
    });
    window.glb.BarFolder.open();
    window.addEventListener('resize', onWindowResize);

    const loader = new window.glb.import_FontLoader();
    loader.load( 'examples/fonts/helvetiker_regular.typeface.json', function ( font ) {
        window.glb.font = font
    })
}

