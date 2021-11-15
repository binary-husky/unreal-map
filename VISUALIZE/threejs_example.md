```python
for index, guard in enumerate(self.guards):
    self.mcv.v2dx('tank|%d|b|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1], guard.state.p_ang, vel_dir=dir2rad(guard.state.p_vel), attack_range=guard.terrain*guard.shootRad)
    if not guard.alive:
        self.mcv.v2dx('tank|%d|k|0.04'%(index), guard.state.p_pos[0], guard.state.p_pos[1], guard.state.p_ang)
for index, attacker in enumerate(self.attackers):
    self.mcv.v2dx('tank|%d|r|0.04'%(index+len(self.guards)), attacker.state.p_pos[0], attacker.state.p_pos[1], attacker.state.p_ang, vel_dir=dir2rad(attacker.state.p_vel), attack_range=attacker.terrain*attacker.shootRad)
    if not attacker.alive:
        self.mcv.v2dx('tank|%d|k|0.04'%(index+len(self.guards)), attacker.state.p_pos[0], attacker.state.p_pos[1], attacker.state.p_ang)
self.mcv.v2d_add_terrain(self.world.init_theta)
self.mcv.v2d_show()
```