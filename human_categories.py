#!/usr/bin/env python
"""human_categories.py

Code to define the class that deals with the specifics
of the 16 categories used in human and DNN experiments.

"""

import numpy as np
import os


def get_human_object_recognition_categories():
    """Return the 16 categories that are used for the human experiment.

    To be more precise, return the categories that Robert uses in his
    object recognition experiment.
    """

    return sorted(["knife", "keyboard", "elephant", "bicycle", "airplane",
            "clock", "oven", "chair", "bear", "boat",
             "car", "bird", "dog", "orange", "refrigerator", "bowl"])


def get_num_human_categories():
    """Return number of categories used in the object recogn. experiment."""

    return len(get_human_object_recognition_categories())


class HumanCategories(object):

    # Note: Some WNIDs may not be part of the ILSVRC 2012 database.

    knife =    ['n03623556', 'n02794368', 'n02864987','n02880842', 'n02893941',
            'n02927053', 'n02973904', 'n02976123', 'n03041632', 'n03235327',
            'n03549473', 'n03658185', 'n03658185', 'n03889397', 'n03890093',
            'n03973628','n04016479', 'n04237287','n04364827','n04380346']

    keyboard = ['n03085013', 'n04505470', 'n03928814', 'n03614007','n04036303']

    elephant = ['n02503517', 'n02504458', 'n02506783', 'n02504770', 'n02503756',
    'n02504013','n02505485', 'n02505238', 'n02505063','n02504196']

    bicycle =  ['n02834778', 'n02835271', 'n03792782', 'n03853924', 'n04026813',
    'n04126066', 'n04524716']

    airplane = ['n02686568', 'n02863638', 'n03140771', 'n03510583', 'n03666917', 'n04308084',
'n02692877', 'n02794972', 'n02850950','n02691156', 'n02690373', 'n02704645', 'n02842573',
'n02867715', 'n03174079', 'n03335030', 'n03490784', 'n03595860', 'n03783873', 'n03798610',
'n04012084', 'n04062644', 'n04160586', 'n04230487', 'n04389999','n02690373', 'n02686121',
'n03809312', 'n04583620','n04308273', 'n03215191' ,'n03577672', 'n03608074', 'n04308397',
 'n03321419', 'n03596543', 'n03604311', 'n04503499','n03227505', 'n04012482', 'n04222723',
 'n03365231', 'n03373611'  ]


    clock =    ['n03046257', 'n02694662', 'n02708093', 'n03027001', 'n03145147',
    'n03196217', 'n03271260', 'n03909406', 'n04378024', 'n04437276', 'n04502059',
    'n04548280', 'n04558347','n02686121', 'n03809312', 'n04583620','n04308273', 'n03215191',
    'n03577672', 'n03608074', 'n04308397','n03321419', 'n03596543', 'n03604311', 'n04503499',
    'n03227505', 'n04012482', 'n04222723','n03365231', 'n03373611']

    oven =     ['n03862676', 'n02905036', 'n03259280', 'n03425241', 'n04111531', 'n04388473']

    chair =    ['n03001627', 'n02738535', 'n02791124', 'n03002210', 'n03002711',
    'n03260849', 'n03335333', 'n03376595', 'n03518445', 'n03632729', 'n03649674',
    'n04099969', 'n04331277', 'n04373704', 'n04381450', 'n04576002','n04593077',
    'n02957862', 'n03262932', 'n03325403', 'n03786621', 'n04062428','n04429376',
    'n02983904', 'n03750540', 'n03802507', 'n03902564','n02946270', 'n03168217', 'n04610176',
    'n02876326', 'n03962932', 'n04201435','n04590933','n02806762', 'n03790953']

    bear =     ['n02131653', 'n01322983', 'n02133161', 'n02133704', 'n02134084',
                'n02134418', 'n02132136', 'n02132320','n02133400','n02132788',
                'n02132466', 'n02132580']

    boat =     ['n02858304', 'n02737351', 'n02792552', 'n02918455', 'n02947660',
     'n03329663', 'n03344393', 'n03447447', 'n03464628', 'n03468570', 'n03603594',
      'n03687820', 'n03696445', 'n03703203', 'n03710079', 'n03790230', 'n03939178',
       'n03977592', 'n04024983', 'n04095210', 'n04150371', 'n04158807', 'n04244997',
       'n04308807', 'n04363671', 'n04409128', 'n04495843','n03236423', 'n03545470',
        'n03981566', 'n04150273', 'n04577293','n02964295','n02932891', 'n03647423',
        'n03859170', 'n04273569', 'n04562122','n03552749','n04208760','n03609786',
        'n04574348', 'n03662601','n02951358', 'n03061345', 'n03105306', 'n03199901',
        'n03436891', 'n04037964', 'n04229480', 'n04612504','n02843029', 'n03254374',
        'n03609235', 'n03861430','n04115456', 'n04577139','n04190997', 'n04038231',
        'n04038338', 'n04156720','n04133114','n03602081','n03436772']

    '''bottle =   ['n04579145','n02823428','n03983396','n03571625',
    'n02962061','n03709363','n03690851','n04246060','n03174450',
    'n02952374','n04560804','n03521675','n04271793','n04591713',
    'n02706221','n03140431','n02985963','n03359566','n02825240',
    'n02960903','n03295246','n03185868','n03937543','n04518132',
    'n03923379','n04579056','n04113968','n04557648','n02876457',
    'n03595409','n03603722','n03813946','n03449451','n04422727']'''

    car = ['n04399269','n04368695','n02704792','n02854630','n03404012','n02701002',
    'n04201733','n02814533','n02924554','n03472937','n03769967','n02930766',
    'n03079136','n03100240','n03119396','n03881534','n03141065','n03268790',
    'n03421669','n03493219','n03498781','n03539103','n03543394','n03594945',
    'n02831335','n03670208','n03680512','n03770085','n03770679','n03777568',
    'n03870105','n03342961','n04322801','n04037443','n04097373','n02907194',
    'n04166281','n04285008','n04285965','n04302988','n04322924','n04347119',
    'n04459122','n04516354','n02958343','n03221643','n03389761','n03444034',
    'n03445924','n03506880','n03785016','n03769722','n04466871','n03790512',
    'n04252225','n03256166','n03632852','n03345487','n03417042','n03690473',
    'n03930630','n04263139','n04461696','n04465666','n04388372','n04467665',
    'n04474035','n02871314','n03173929','n03648667','n03764822','n03884639',
    'n03796401','n03896419','n03977966','n04520170','n04490091']

    bird = ['n02055803','n01503976','n01514752','n01514668','n01514859','n01514926',
    'n01515217','n01515078','n01515303','n01515583','n01516212','n01516609',
    'n01517036','n01517389','n01518878','n01519563','n01519873','n01520576',
    'n01521399','n01521756','n01522450','n01523248','n01523493','n01523105',
    'n01517565','n01517966','n01526521','n01526766','n01527347','n01527194',
    'n01527917','n01527617','n01528396','n01528845','n01528654','n01530439',
    'n01530575','n01531178','n01531344','n01531512','n01531639','n01531811',
    'n01531971','n01532325','n01532511','n01532829','n01533000','n01533481',
    'n01533339','n01533651','n01533893','n01534155','n01534582','n01534433',
    'n01535140','n01535469','n01535690','n01536035','n01536186','n01536334',
    'n01536644','n01536780','n01534762','n01537544','n01537895','n01538059',
    'n01538200','n01538362','n01538630','n01537134','n01540566','n01540832',
    'n01541102','n01540233','n01541386','n01541760','n01542168','n01542433',
    'n01541922','n01545010','n01544704','n01529672','n01539272','n01538955',
    'n01543175','n01543383','n01543632','n01543936','n01544389','n01544208',
    'n01542786','n01556182','n01556514','n01555809','n01557962','n01558149',
    'n01558307','n01558461','n01558594','n01558765','n01558993','n01559160',
    'n01559477','n01559639','n01559804','n01560419','n01560105','n01560280',
    'n01560793','n01560935','n01560636','n01561181','n01561452','n01561732',
    'n01562014','n01562265','n01562451','n01557185','n01563449','n01563945',
    'n01564101','n01564217','n01563746','n01564773','n01565345','n01565599',
    'n01565930','n01566207','n01564394','n01564914','n01565078','n01567678',
    'n01567879','n01568294','n01568132','n01568720','n01568892','n01569060',
    'n01569262','n01569423','n01569566','n01569971','n01569836','n01570267',
    'n01570421','n01570839','n01570676','n01567133','n01563128','n01566645',
    'n01571410','n01571126','n01572489','n01572654','n01572328','n01572782',
    'n01573240','n01573360','n01573074','n01573627','n01573898','n01574560',
    'n01574390','n01574801','n01575117','n01575401','n01574045','n01571904',
    'n01576076','n01576358','n01575745','n01577035','n01577458','n01577941',
    'n01578180','n01577659','n01576695','n01579149','n01579028','n01579260',
    'n01579410','n01579578','n01579729','n01580490','n01580379','n01580870',
    'n01580772','n01581434','n01581166','n01580077','n01581874','n01581984',
    'n01581730','n01582398','n01582498','n01582220','n01578575','n01583209',
    'n01583495','n01583828','n01582856','n01586941','n01587278','n01587526',
    'n01588002','n01587834','n01588725','n01588996','n01588431','n01589718',
    'n01589893','n01590220','n01589286','n01591005','n01591123','n01591301',
    'n01590583','n01592257','n01592540','n01592084','n01592387','n01592694',
    'n01593028','n01593282','n01593553','n01591697','n01594004','n01594787',
    'n01594968','n01595168','n01595450','n01595974','n01596273','n01596608',
    'n01595624','n01594372','n01597022','n01597737','n01597906','n01598074',
    'n01598271','n01597336','n01599159','n01599269','n01599388','n01598988',
    'n01599556','n01599741','n01600341','n01600085','n01598588','n01601068',
    'n01601410','n01600657','n01602080','n01602209','n01601694','n01602832',
    'n01603000','n01603152','n01602630','n01603812','n01603953','n01603600',
    'n01525720','n01539925','n01540090','n01539573','n01545574','n01546039',
    'n01546506','n01548492','n01548694','n01548865','n01549053','n01548301',
    'n01549641','n01549430','n01549886','n01550172','n01551080','n01551300',
    'n01552034','n01552333','n01550761','n01555305','n01547832','n01551711',
    'n01552813','n01553527','n01553762','n01554017','n01553142','n01554448',
    'n01555004','n01546921','n01584695','n01584853','n01585287','n01585422',
    'n01585121','n01585715','n01586020','n01586374','n01584225','n01524359',
    'n01524761','n01604968','n01606097','n01606177','n01606522','n01606672',
    'n01606809','n01606978','n01607309','n01607429','n01607600','n01607812',
    'n01607962','n01608265','n01608814','n01609062','n01609391','n01608432',
    'n01609956','n01610100','n01610226','n01609751','n01610552','n01611674',
    'n01611472','n01611800','n01611969','n01612122','n01612275','n01612476',
    'n01612955','n01613177','n01612628','n01610955','n01616086','n01605630',
    'n01613807','n01614038','n01614690','n01614343','n01614556','n01614925',
    'n01615303','n01615458','n01615703','n01615121','n01613294','n01616551',
    'n01617095','n01617443','n01617766','n01618082','n01616764','n01619310',
    'n01619835','n01620135','n01619536','n01620414','n01620735','n01618922',
    'n01616318','n01618503','n01621635','n01622120','n01622483','n01622352',
    'n01622779','n01622959','n01623110','n01623425','n01623615','n01624115',
    'n01624212','n01623706','n01623880','n01624305','n01624537','n01624833',
    'n01625121','n01625562','n01621127','n01604330','n01790171','n01790304',
    'n01790398','n01790557','n01790711','n01790812','n01792042','n01792429',
    'n01792158','n01792530','n01792808','n01792955','n01793085','n01793159',
    'n01793249','n01792640','n01793340','n01793435','n01793565','n01793715',
    'n01791625','n01791954','n01794344','n01794158','n01809371','n01809106',
    'n01789740','n01791314','n01791388','n01791463','n01791107','n01794651',
    'n01800195','n01799302','n01799679','n01800633','n01800424','n01801672',
    'n01801479','n01801876','n01802159','n01801088','n01809752','n01811243',
    'n01811542','n01812187','n01813532','n01813658','n01813385','n01813948',
    'n01814217','n01812337','n01812662','n01812866','n01813088','n01814620',
    'n01814755','n01815036','n01814921','n01814370','n01814549','n01815270',
    'n01811909','n01816017','n01816140','n01816474','n01815601','n01810700',
    'n01795735','n01795900','n01796019','n01796105','n01795545','n01796729',
    'n01796800','n01796519','n01796340','n01797020','n01797307','n01797601',
    'n01797886','n01798168','n01798706','n01798839','n01798979','n01798484',
    'n01795088','n01803362','n01803641','n01803893','n01804163','n01805321',
    'n01806061','n01806143','n01806297','n01806364','n01806467','n01805801',
    'n01807105','n01803078','n01804653','n01804478','n01805070','n01804921',
    'n01806847','n01806567','n01807828','n01808140','n01808291','n01808596',
    'n01807496','n01802721','n01810268','n02153203','n01789386','n01817263',
    'n01817346','n01817953','n01818299','n01818515','n01818832','n01819313',
    'n01819465','n01819115','n01819734','n01820052','n01820801','n01821076',
    'n01820546','n01820348','n01821554','n01821869','n01822300','n01821203',
    'n01816887','n01823414','n01823740','n01824035','n01824344','n01824749',
    'n01824862','n01824575','n01823013','n01825278','n01822602','n01826680',
    'n01826844','n01826364','n01827793','n01828096','n01828556','n01827403',
    'n01828970','n01829413','n01830042','n01829869','n01830479','n01830915',
    'n01831360','n01825930','n01832493','n01832813','n01833112','n01832167',
    'n01833415','n01834177','n01834540','n01833805','n01831712','n01835769',
    'n01835918','n01836087','n01836384','n01836673','n01835276','n01837072',
    'n01837526','n01834918','n01839086','n01839330','n01839750','n01839949',
    'n01840120','n01839598','n01840412','n01840775','n01841288','n01841441',
    'n01841102','n01841679','n01841943','n01838598','n01842235','n01842504',
    'n01842788','n01843065','n01843719','n01843383','n01838038','n01844746',
    'n01844551','n01844231','n01847000','n01847089','n01847170','n01847253',
    'n01847407','n01847806','n01847978','n01848323','n01848453','n01848555',
    'n01848123','n01848840','n01848648','n01848976','n01849157','n01849676',
    'n01849466','n01849863','n01850192','n01850553','n01850373','n01850873',
    'n01851038','n01851207','n01851573','n01851731','n01851375','n01851895',
    'n01852329','n01852142','n01852400','n01852671','n01853195','n01853666',
    'n01853498','n01853870','n01854700','n01854838','n01855032','n01855188',
    'n01855476','n01854415','n01852861','n01846331','n01856072','n01856155',
    'n01856380','n01856553','n01857079','n01856890','n01857512','n01857325',
    'n01857632','n01857851','n01855672','n01845477','n01860864','n01861330',
    'n01861148','n01860497','n01845132','n01858281','n01858780','n01858845',
    'n01858906','n01859190','n01859325','n01859689','n01859852','n01859496',
    'n01860002','n01860187','n01858441','n02002556','n02002724','n02003037',
    'n02003204','n02003577','n02003839','n02004131','n02004492','n02004855',
    'n02002075','n02005399','n02006063','n02006364','n02005790','n02006985',
    'n02007284','n02006656','n02007558','n02008497','n02008643','n02009380',
    'n02009508','n02009750','n02009912','n02010272','n02008796','n02009229',
    'n02010728','n02011016','n02010453','n02011281','n02011805','n02011943',
    'n02012185','n02011460','n02008041','n02013177','n02012849','n02013567',
    'n02013706','n02014237','n02014524','n02015357','n02015797','n02016066',
    'n02015554','n02017725','n02018207','n02018368','n02018027','n02014941',
    'n02019190','n02019438','n02018795','n02020219','n02019929','n02020345',
    'n02020578','n02021281','n02021050','n02023855','n02023992','n02024185',
    'n02024479','n02024763','n02025239','n02025389','n02025043','n02023341',
    'n02026948','n02027075','n02027357','n02027492','n02027897','n02028035',
    'n02028342','n02028451','n02028175','n02028727','n02028900','n02029087',
    'n02029378','n02029706','n02030224','n02030035','n02030568','n02030837',
    'n02030287','n02026059','n02026629','n02031298','n02031585','n02030996',
    'n02032222','n02032355','n02032480','n02032769','n02033208','n02033324',
    'n02033041','n02031934','n02033779','n02033882','n02033561','n02034295',
    'n02034129','n02034971','n02035210','n02035402','n02035656','n02034661',
    'n02036228','n02036053','n02036711','n02037110','n02037869','n02038141',
    'n02038466','n02037464','n02038993','n02039497','n02039780','n02039171',
    'n02040266','n02022684','n02000954','n02016659','n02016816','n02017213',
    'n02017475','n02016956','n02016358','n02041678','n02041875','n02042046',
    'n02042180','n02042472','n02042759','n02041246','n02043333','n02043063',
    'n02041085','n02043808','n02044517','n02044908','n02044778','n02044178',
    'n02040505','n02045596','n02045864','n02046171','n02046442','n02046939',
    'n02047045','n02047411','n02047517','n02047260','n02046759','n02045369',
    'n02047975','n02048115','n02048353','n02047614','n02049088','n02048698',
    'n02050313','n02050442','n02050586','n02050809','n02051059','n02050004',
    'n02049532','n02052204','n02052365','n02051845','n02052775','n02053425',
    'n02053584','n02053083','n02054036','n02054711','n02054502','n02055107',
    'n02051474','n02056228','n02056570','n02056728','n02057035','n02057330',
    'n02055658','n02057898','n02058594','n02058747','n02058221','n02059541',
    'n02059852','n02060133','n02060569','n02060411','n02061217','n02061560',
    'n02060889','n02059162','n02061853','n02057731','n02021795','n01844917','n02511730']

    dog = ["n02100583","n02100236","n02100735","n02101006","n02100877",
    "n02098806","n02101670","n02102806","n02102973", "n02102177","n02102040",
    "n02101556","n02101388","n02102318", "n02102480","n02103181","n02098906",
    "n02099601","n02099849","n02099429","n02099267","n02099712","n02089468",
    "n02096294","n02095212","n02095314","n02098105","n02095889","n02095570",
    "n02096437","n02096051","n02098413","n02094433","n02098286","n02097130",
    "n02097047","n02097209","n02093256","n02093428","n02094114","n02096177",
    "n02093859","n02097298","n02096585","n02093647","n02093991","n02097658",
    "n02097967","n02094258","n02097474","n02094931","n02093754","n02087314",
    "n02090253","n02090622","n02090721","n02092002","n02089078","n02088992",
    "n02089867","n02089725","n02089973","n02092339","n02091635","n02088466",
    "n02091467","n02091831","n02088094","n02092173","n02091134","n02091032",
    "n02088364","n02090129","n02088238","n02088632","n02090379","n02088745",
    "n02091244","n02087394","n02110532","n02085118","n02085019","n02113186",
    "n02113023","n02113978","n02085272","n02111277","n02113712","n02113892",
    "n02113624","n02113799","n02110806","n02111129","n02112706","n02110958",
    "n02109047","n02109256","n02105641","n02106382","n02106550","n02105505",
    "n02106030","n02106166","n02105162","n02105056","n02105855","n02105412",
    "n02105251","n02106662","n02104365","n02107142","n02110627","n02107312",
    "n02104029","n02104280","n02104184","n02106854","n02109687","n02110185",
    "n02110063","n02108089","n02108422","n02109961","n02108000","n02107683",
    "n02107574","n02107908","n02109525","n02108551","n02109391","n02108915",
    "n02112018","n02112350","n02112137","n02111889","n02111500","n02086753",
    "n02086910","n02086646","n02086079","n02085936","n02087046","n02085782",
    "n02086240","n02085620","n01322604"]

    orange = ['n07747607','n07747811', 'n07748753','n07748912']

    refrigerator = ['n04070727','n03102654','n03273913']

    bowl =     ['n02881397','n04263257','n03775546', 'n02880940','n04023695', 'n02997910',
     'n04242704','n04263257', 'n03775546', 'n04130257', 'n03341606', 'n03984759']


    def get_human_category_from_WNID(self, wnid):
        """Return the MS COCO category for a given WNID.

        Returns None if wnid is not part of the 16 human categories.

        parameters:
        - wnid: a string containing the wnid of an image, e.g. 'n03658185'

        """

        categories = get_human_object_recognition_categories()
        for c in categories:
            attr = getattr(self, c)
            if wnid in attr:
                return c

        return None
