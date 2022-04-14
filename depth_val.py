import numpy as np
from sklearn.linear_model import LinearRegression
FPS=25
depth=[0.49175581336021423, 0.5224343538284302, 0.5726707577705383, 0.5046202540397644, 0.4913043677806854, 0.47142237424850464, 0.46519774198532104, 0.44720757007598877, 0.42353498935699463, 0.4016455411911011, 0.3367178738117218, 0.3306635022163391, 0.32438814640045166, 0.29709285497665405, 0.35044634342193604, 0.31084340810775757, 0.34751662611961365, 0.3224572539329529, 0.3014606535434723, 0.35453924536705017, 0.3379848599433899, 0.3369865119457245, 0.3152981102466583, 0.3423975110054016, 0.3170645534992218, 0.28820204734802246, 0.35262107849121094, 0.35454967617988586, 0.2828901410102844, 0.29982054233551025, 0.28612303733825684, 0.28843024373054504, 0.28586214780807495, 0.3092862367630005, 0.3222945034503937, 0.3789243996143341, 0.39875149726867676, 0.3592897355556488, 0.34230709075927734, 0.3683692216873169, 0.3605717420578003, 0.3677347004413605, 0.4266679883003235, 0.37972861528396606, 0.3867117762565613, 0.36677441000938416, 0.334412157535553, 0.3080103099346161, 0.2901912331581116, 0.29657799005508423, 0.27134039998054504, 0.2608604431152344, 0.3000936210155487, 0.30887657403945923, 0.3219429552555084, 0.3146926462650299, 0.2735424041748047, 0.2830522954463959, 0.24527335166931152, 0.2914275527000427, 0.3849026560783386, 0.29411473870277405, 0.35732582211494446, 0.3398340344429016, 0.3035910427570343, 0.34690430760383606, 0.3194737732410431, 0.3085327446460724, 0.28210777044296265, 0.2364283949136734, 0.28880125284194946, 0.2501901090145111, 0.29056140780448914, 0.32213225960731506, 0.2862406373023987, 0.2638574242591858, 0.2617970108985901, 0.23887450993061066, 0.2294398546218872, 0.27422863245010376, 0.3485122621059418, 0.34715867042541504, 0.2818213999271393, 0.3043860197067261, 0.2977524399757385, 0.27095940709114075, 0.2250656634569168, 0.20730449259281158, 0.17763826251029968, 0.1971241980791092, 0.18886946141719818, 0.1755131036043167, 0.1995922029018402, 0.22185425460338593, 0.29832515120506287, 0.2965415120124817, 0.20498698949813843, 0.17982855439186096, 0.21281041204929352, 0.20117315649986267, 0.25811824202537537, 0.3001033365726471, 0.2151133120059967, 0.21217599511146545, 0.2112300992012024, 0.27576714754104614, 0.20980221033096313, 0.18210722506046295, 0.17265279591083527, 0.18577510118484497, 0.08424507826566696, 0.10031836479902267, 0.09757832437753677, 0.17350497841835022, 0.11241647601127625, 0.19129818677902222, 0.13509373366832733, 0.0963590145111084, 0.14516322314739227, 0.1397634744644165, 0.06131741404533386, 0.04298953339457512, 0.08059637248516083, 0.057699672877788544, 0.10732371360063553, 0.09248562902212143, 0.10556823760271072, 0.060839492827653885, 0.07810088247060776, 0.11278342455625534, 0.1088719293475151, 0.17106598615646362, 0.13655465841293335, 0.12628063559532166, 0.09458687901496887, 0.10144763439893723, 0.04549681022763252, 0.04365085810422897, 0.053576644510030746, 0.06424552202224731, 0.030591165646910667, 0.08448520302772522, 0.08592232316732407, 0.0822213739156723, 0.1475072205066681, 0.13092994689941406, 0.07896748185157776, 0.08059900999069214, 0.05433450639247894, 0.032344382256269455, 0.08166404068470001, 0.11007364839315414, 0.11043640226125717, 0.12057295441627502, 0.07800356298685074, 0.0961969643831253, 0.025690389797091484, 0.030860913917422295, 0.007885572500526905, -0.01923399232327938, 0.09141870588064194, 0.16388633847236633, 0.18817760050296783, 0.1964707374572754, 0.15117500722408295, 0.09132107347249985, 0.12158635258674622, 0.10448332130908966, 0.08011388033628464, 0.09979713708162308, 0.1013718843460083, 0.14941896498203278, 0.15974673628807068, 0.10181449353694916, 0.13306963443756104, 0.115974560379982, 0.0994001179933548, 0.10433328151702881, 0.04168153181672096, 0.08025045692920685, 0.036506734788417816, 0.05730769410729408, 0.07306748628616333, 0.08126690983772278, 0.034479767084121704, 0.049922846257686615, 0.07997425645589828, 0.1189674437046051, 0.09057189524173737, 0.09053169935941696, 0.08457102626562119, 0.09843344241380692, 0.08017352223396301, 0.06789441406726837, 0.12152837961912155, 0.10915343463420868, 0.08935751020908356, 0.1357744336128235, 0.15005844831466675, 0.1111178994178772, 0.07236313819885254, 0.14946205914020538, 0.0950607880949974, 0.12228072434663773, 0.09918656200170517, 0.10168270766735077, 0.10393218696117401, 0.1618744134902954, 0.16297173500061035, 0.1821230798959732, 0.1455201804637909, 0.12508638203144073, 0.10193618386983871, 0.08106256276369095, 0.1039847880601883, 0.09905959665775299, 0.16585545241832733, 0.13115838170051575, 0.08171099424362183, 0.06325089186429977, 0.06860372424125671, 0.07831276953220367, 0.00133819121401757, 0.06818629801273346, 0.059257641434669495, 0.04780000075697899, 0.04857571795582771, 0.06750613451004028]
depths=[depth[38*i:38*(i+1)] for i in range(int(len(depth)/38))]
x=np.arange(38).reshape(-1,1)
print("\n\n STARTING THE PROGRAM \n\n\n")
for i in range(6):
    
    depth_bucket=np.array(depths[i])
    print(depth_bucket)
    lr=LinearRegression()
    model=lr.fit(x,depth_bucket)
    depth_end=list(model.predict(np.array([0,37]).reshape(-1,1)))
    print(depth_end)
    delta_depth=(depth_end[1]-depth_end[0])
    avg_depth=(depth_end[1]+depth_end[0])/2
    vel_x=delta_depth*FPS*avg_depth
    print(vel_x)
    print("\n\n\n\n")

#vel_y=((tracks[i][0][0]-tracks[i][37][0])+(tracks[i][0][2]-tracks[i][37][2]))*FPS/2560
#each_item["velocity"].append((vel_x,vel_y))