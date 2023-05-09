import numpy as np
import rasterio
import math
import re
import warnings
import cv2
from pyproj import CRS
from pathlib import Path

EQUATOR_ARC_SECOND_IN_METERS = 30.87  # meters

raster_metadata = {  # res in arcsec
    'NavWater1994_WGS84': {'min_val': 0.0, 'max_val': 4.0, 'nan': -9999.0, 'new_nan': -1, 'mu': 0.06879497, 'sigma': 0.411931725, 'res': 35.613844445368, 'cat':None},
    'NavWater2009_WGS84': {'min_val': 0.0, 'max_val': 4.0, 'nan': -9999.0, 'new_nan': -1, 'mu': 0.07914783, 'sigma': 0.44055015, 'res': 35.613844445368, 'cat':None},
    'Railways_WGS84': {'min_val': 0.0, 'max_val': 8.0, 'nan': 255, 'new_nan': -1, 'mu': 0.15251939, 'sigma': 1.0940273, 'res': 35.613844445368, 'cat':[8.0]},
    'Roads_WGS84': {'min_val': 0.0, 'max_val': 8.0, 'nan': -2, 'new_nan': -1, 'mu': 1.3234715, 'sigma': 2.2345204, 'res': 35.613844445368, 'cat':None}, # nan = everything negative
    'Pasture2009_WGS84': {'min_val': 0.0, 'max_val': 4.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.4739912, 'sigma': 0.9447189, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0]},
    'Pasture1993_WGS84': {'min_val': 0.0, 'max_val': 4.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.50720686, 'sigma': 0.9687058, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0]},
    'croplands2005_WGS84': {'min_val': 0.0, 'max_val': 7.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.95726395, 'sigma': 2.4050977, 'res': 35.613844445368, 'cat':[7.0]},
    'croplands1992_WGS84': {'min_val': 0.0, 'max_val': 7.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.7948161, 'sigma': 2.2208056, 'res': 35.613844445368, 'cat':[7.0]},
    'Lights2009_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.36079723, 'sigma': 1.5246228, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]},
    'Lights1994_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 127.0, 'new_nan': -1, 'mu': 0.29228857, 'sigma': 1.4004335, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]},
    'Popdensity2010_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 2147483600.0, 'new_nan': -1, 'mu': 2.3217769, 'sigma': 2.5324972, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]},
    'Popdensity1990_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 127.0, 'new_nan': -1, 'mu': 2.100136, 'sigma': 2.3988833, 'res': 35.613844445368, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]},
    'Built2009_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 2.14748365e+09, 'new_nan': -1, 'mu': 0.18676297, 'sigma': 1.3537824, 'res': 35.613844445368, 'cat':[10.0]},
    'Built1994_WGS84': {'min_val': 0.0, 'max_val': 10.0, 'nan': 2.14748365e+09, 'new_nan': -1, 'mu': 0.16713549, 'sigma': 1.281962, 'res': 35.613844445368, 'cat':[10.0]},
    'biomes01': {'min_val': 1.0, 'max_val': 14.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 7.820812352118446, 'sigma': 3.8276948440061425, 'res': 360, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]},
    'biomes001': {'min_val': 1.0, 'max_val': 14.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 7.820618402801339, 'sigma': 3.8274062735774326, 'res': 36, 'cat':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]},
    'ecoregions01': {'min_val': 10101.0, 'max_val': 81333.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 57775.340511425624, 'sigma': 23246.668162355003, 'res': 360,
                     'cat':[10101.0, 10102.0, 10103.0, 10104.0, 10105.0, 10106.0, 10107.0, 10108.0, 10110.0, 10111.0, 10112.0, 10113.0, 10114.0, 10115.0, 10116.0, 10117.0, 10118.0, 10119.0, 10120.0, 10121.0, 10122.0, 10123.0, 10124.0, 10125.0, 10126.0, 10127.0, 10128.0, 10201.0, 10202.0, 10203.0, 10204.0, 10401.0, 10402.0, 10403.0, 10404.0, 10405.0, 10406.0, 10407.0, 10408.0, 10409.0, 10410.0, 10411.0, 10412.0, 10413.0, 10414.0, 10701.0, 10702.0, 10703.0, 10704.0, 10705.0, 10706.0, 10707.0, 10708.0, 10709.0, 10801.0, 10802.0, 10803.0, 11001.0, 11002.0, 11003.0, 11101.0, 11201.0, 11202.0, 11203.0, 11204.0, 11205.0, 11206.0, 11207.0, 11208.0, 11209.0, 11210.0, 11301.0, 11302.0, 11303.0, 11304.0, 11305.0, 11306.0, 11307.0, 11308.0, 11309.0, 11310.0, 11401.0, 21101.0, 21102.0, 21103.0, 21104.0, 30101.0, 30102.0, 30103.0, 30104.0, 30105.0, 30106.0, 30107.0, 30108.0, 30109.0, 30110.0, 30111.0, 30112.0, 30113.0, 30114.0, 30115.0, 30116.0, 30117.0, 30118.0, 30119.0, 30120.0, 30121.0, 30122.0, 30123.0, 30124.0, 30125.0, 30126.0, 30127.0, 30128.0, 30129.0, 30130.0, 30201.0, 30202.0, 30203.0, 30701.0, 30702.0, 30703.0, 30704.0, 30705.0, 30706.0, 30707.0, 30708.0, 30709.0, 30710.0, 30711.0, 30712.0, 30713.0, 30714.0, 30715.0, 30716.0, 30717.0, 30718.0, 30719.0, 30720.0, 30721.0, 30722.0, 30723.0, 30724.0, 30725.0, 30726.0, 30801.0, 30802.0, 30803.0, 30901.0, 30902.0, 30903.0, 30904.0, 30905.0, 30906.0, 30907.0, 30908.0, 31001.0, 31002.0, 31003.0, 31004.0, 31005.0, 31006.0, 31007.0, 31008.0, 31009.0, 31010.0, 31011.0, 31012.0, 31013.0, 31014.0, 31015.0, 31201.0, 31202.0, 31203.0, 31301.0, 31302.0, 31303.0, 31304.0, 31305.0, 31306.0, 31307.0, 31308.0, 31309.0, 31310.0, 31311.0, 31312.0, 31313.0, 31314.0, 31315.0, 31316.0, 31318.0, 31319.0, 31320.0, 31321.0, 31322.0, 31401.0, 31402.0, 31403.0, 31404.0, 31405.0, 40101.0, 40102.0, 40103.0, 40104.0, 40105.0, 40106.0, 40107.0, 40108.0, 40109.0, 40110.0, 40111.0, 40112.0, 40113.0, 40114.0, 40115.0, 40116.0, 40117.0, 40118.0, 40119.0, 40120.0, 40121.0, 40122.0, 40123.0, 40124.0, 40125.0, 40126.0, 40127.0, 40128.0, 40129.0, 40130.0, 40131.0, 40132.0, 40133.0, 40134.0, 40135.0, 40136.0, 40137.0, 40138.0, 40139.0, 40140.0, 40141.0, 40142.0, 40143.0, 40144.0, 40145.0, 40146.0, 40147.0, 40148.0, 40149.0, 40150.0, 40151.0, 40152.0, 40153.0, 40154.0, 40155.0, 40156.0, 40157.0, 40158.0, 40159.0, 40160.0, 40161.0, 40162.0, 40163.0, 40164.0, 40165.0, 40166.0, 40167.0, 40168.0, 40169.0, 40170.0, 40171.0, 40172.0, 40201.0, 40202.0, 40203.0, 40204.0, 40205.0, 40206.0, 40207.0, 40208.0, 40209.0, 40210.0, 40211.0, 40212.0, 40301.0, 40302.0, 40303.0, 40304.0, 40401.0, 40402.0, 40403.0, 40501.0, 40502.0, 40701.0, 40901.0, 41001.0, 41301.0, 41302.0, 41303.0, 41304.0, 41401.0, 41402.0, 41403.0, 41404.0, 41405.0, 41406.0, 50201.0, 50302.0, 50303.0, 50401.0, 50402.0, 50403.0, 50404.0, 50405.0, 50406.0, 50407.0, 50408.0, 50409.0, 50410.0, 50411.0, 50412.0, 50413.0, 50414.0, 50415.0, 50416.0, 50417.0, 50501.0, 50502.0, 50503.0, 50504.0, 50505.0, 50506.0, 50507.0, 50508.0, 50509.0, 50510.0, 50511.0, 50512.0, 50513.0, 50514.0, 50515.0, 50516.0, 50517.0, 50518.0, 50519.0, 50520.0, 50521.0, 50522.0, 50523.0, 50524.0, 50525.0, 50526.0, 50527.0, 50528.0, 50529.0, 50530.0, 50601.0, 50602.0, 50603.0, 50604.0, 50605.0, 50606.0, 50607.0, 50608.0, 50609.0, 50610.0, 50611.0, 50612.0, 50613.0, 50614.0, 50615.0, 50616.0, 50617.0, 50701.0, 50801.0, 50802.0, 50803.0, 50804.0, 50805.0, 50806.0, 50807.0, 50808.0, 50809.0, 50810.0, 50811.0, 50812.0, 50813.0, 50814.0, 50815.0, 51101.0, 51102.0, 51103.0, 51104.0, 51105.0, 51106.0, 51107.0, 51108.0, 51109.0, 51110.0, 51111.0, 51112.0, 51113.0, 51114.0, 51115.0, 51116.0, 51117.0, 51118.0, 51201.0, 51202.0, 51203.0, 51301.0, 51302.0, 51303.0, 51304.0, 51305.0, 51306.0, 51307.0, 51308.0, 51309.0, 51310.0, 51311.0, 51312.0, 51313.0, 60101.0, 60102.0, 60103.0, 60104.0, 60105.0, 60106.0, 60107.0, 60108.0, 60109.0, 60111.0, 60112.0, 60113.0, 60114.0, 60115.0, 60116.0, 60117.0, 60118.0, 60119.0, 60120.0, 60121.0, 60122.0, 60124.0, 60125.0, 60126.0, 60127.0, 60128.0, 60129.0, 60130.0, 60131.0, 60132.0, 60133.0, 60134.0, 60135.0, 60136.0, 60137.0, 60138.0, 60139.0, 60140.0, 60141.0, 60142.0, 60143.0, 60144.0, 60145.0, 60146.0, 60147.0, 60148.0, 60149.0, 60150.0, 60151.0, 60152.0, 60153.0, 60154.0, 60155.0, 60156.0, 60157.0, 60158.0, 60159.0, 60160.0, 60161.0, 60162.0, 60163.0, 60164.0, 60165.0, 60166.0, 60167.0, 60168.0, 60169.0, 60170.0, 60171.0, 60173.0, 60174.0, 60175.0, 60176.0, 60177.0, 60178.0, 60179.0, 60180.0, 60181.0, 60182.0, 60201.0, 60202.0, 60204.0, 60205.0, 60206.0, 60207.0, 60209.0, 60210.0, 60211.0, 60212.0, 60213.0, 60214.0, 60215.0, 60216.0, 60217.0, 60218.0, 60219.0, 60220.0, 60221.0, 60222.0, 60223.0, 60224.0, 60225.0, 60226.0, 60227.0, 60228.0, 60229.0, 60230.0, 60232.0, 60233.0, 60235.0, 60301.0, 60302.0, 60303.0, 60304.0, 60305.0, 60306.0, 60307.0, 60308.0, 60309.0, 60310.0, 60401.0, 60402.0, 60404.0, 60702.0, 60703.0, 60704.0, 60707.0, 60708.0, 60709.0, 60710.0, 60801.0, 60802.0, 60803.0, 60805.0, 60902.0, 60903.0, 60904.0, 60905.0, 60906.0, 60907.0, 60908.0, 60909.0, 61001.0, 61002.0, 61003.0, 61004.0, 61005.0, 61006.0, 61007.0, 61008.0, 61010.0, 61201.0, 61301.0, 61303.0, 61304.0, 61305.0, 61306.0, 61307.0, 61308.0, 61309.0, 61312.0, 61313.0, 61314.0, 61315.0, 61316.0, 61401.0, 61402.0, 61403.0, 61404.0, 61405.0, 61406.0, 61407.0, 70101.0, 70102.0, 70103.0, 70104.0, 70105.0, 70106.0, 70107.0, 70108.0, 70109.0, 70110.0, 70111.0, 70112.0, 70113.0, 70114.0, 70115.0, 70116.0, 70117.0, 70201.0, 70202.0, 70203.0, 70204.0, 70701.0, 70702.0, 80101.0, 80102.0, 80401.0, 80402.0, 80403.0, 80404.0, 80405.0, 80406.0, 80407.0, 80408.0, 80409.0, 80410.0, 80411.0, 80412.0, 80413.0, 80414.0, 80415.0, 80416.0, 80417.0, 80418.0, 80419.0, 80420.0, 80421.0, 80422.0, 80423.0, 80424.0, 80425.0, 80426.0, 80427.0, 80428.0, 80429.0, 80430.0, 80431.0, 80432.0, 80433.0, 80434.0, 80435.0, 80436.0, 80437.0, 80438.0, 80439.0, 80440.0, 80441.0, 80442.0, 80443.0, 80444.0, 80445.0, 80446.0, 80501.0, 80502.0, 80503.0, 80504.0, 80505.0, 80506.0, 80507.0, 80508.0, 80509.0, 80510.0, 80511.0, 80512.0, 80513.0, 80514.0, 80515.0, 80516.0, 80517.0, 80518.0, 80519.0, 80520.0, 80521.0, 80601.0, 80602.0, 80603.0, 80604.0, 80605.0, 80606.0, 80607.0, 80608.0, 80609.0, 80610.0, 80611.0, 80801.0, 80802.0, 80803.0, 80804.0, 80805.0, 80806.0, 80807.0, 80808.0, 80809.0, 80810.0, 80811.0, 80812.0, 80813.0, 80814.0, 80815.0, 80816.0, 80817.0, 80818.0, 80901.0, 80902.0, 80903.0, 80904.0, 80905.0, 80906.0, 80907.0, 80908.0, 81001.0, 81002.0, 81003.0, 81004.0, 81005.0, 81006.0, 81007.0, 81008.0, 81009.0, 81010.0, 81011.0, 81012.0, 81013.0, 81014.0, 81015.0, 81016.0, 81017.0, 81018.0, 81019.0, 81020.0, 81021.0, 81022.0, 81101.0, 81102.0, 81103.0, 81104.0, 81105.0, 81106.0, 81107.0, 81108.0, 81109.0, 81110.0, 81111.0, 81112.0, 81113.0, 81114.0, 81201.0, 81202.0, 81203.0, 81204.0, 81205.0, 81206.0, 81207.0, 81208.0, 81209.0, 81210.0, 81211.0, 81212.0, 81213.0, 81214.0, 81215.0, 81216.0, 81217.0, 81218.0, 81219.0, 81220.0, 81221.0, 81222.0, 81301.0, 81302.0, 81303.0, 81304.0, 81305.0, 81306.0, 81307.0, 81308.0, 81309.0, 81310.0, 81311.0, 81312.0, 81313.0, 81314.0, 81315.0, 81316.0, 81317.0, 81318.0, 81319.0, 81320.0, 81321.0, 81322.0, 81323.0, 81324.0, 81325.0, 81326.0, 81327.0, 81328.0, 81329.0, 81330.0, 81331.0, 81332.0, 81333.0]
    },
    'ecoregions001': {'min_val': 10101.0, 'max_val': 81333.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 57774.163653140575, 'sigma': 23247.29986839079, 'res': 36,
                     'cat':[10101.0, 10102.0, 10103.0, 10104.0, 10105.0, 10106.0, 10107.0, 10108.0, 10110.0, 10111.0, 10112.0, 10113.0, 10114.0, 10115.0, 10116.0, 10117.0, 10118.0, 10119.0, 10120.0, 10121.0, 10122.0, 10123.0, 10124.0, 10125.0, 10126.0, 10127.0, 10128.0, 10201.0, 10202.0, 10203.0, 10204.0, 10401.0, 10402.0, 10403.0, 10404.0, 10405.0, 10406.0, 10407.0, 10408.0, 10409.0, 10410.0, 10411.0, 10412.0, 10413.0, 10414.0, 10701.0, 10702.0, 10703.0, 10704.0, 10705.0, 10706.0, 10707.0, 10708.0, 10709.0, 10801.0, 10802.0, 10803.0, 11001.0, 11002.0, 11003.0, 11101.0, 11201.0, 11202.0, 11203.0, 11204.0, 11205.0, 11206.0, 11207.0, 11208.0, 11209.0, 11210.0, 11301.0, 11302.0, 11303.0, 11304.0, 11305.0, 11306.0, 11307.0, 11308.0, 11309.0, 11310.0, 11401.0, 21101.0, 21102.0, 21103.0, 21104.0, 30101.0, 30102.0, 30103.0, 30104.0, 30105.0, 30106.0, 30107.0, 30108.0, 30109.0, 30110.0, 30111.0, 30112.0, 30113.0, 30114.0, 30115.0, 30116.0, 30117.0, 30118.0, 30119.0, 30120.0, 30121.0, 30122.0, 30123.0, 30124.0, 30125.0, 30126.0, 30127.0, 30128.0, 30129.0, 30130.0, 30201.0, 30202.0, 30203.0, 30701.0, 30702.0, 30703.0, 30704.0, 30705.0, 30706.0, 30707.0, 30708.0, 30709.0, 30710.0, 30711.0, 30712.0, 30713.0, 30714.0, 30715.0, 30716.0, 30717.0, 30718.0, 30719.0, 30720.0, 30721.0, 30722.0, 30723.0, 30724.0, 30725.0, 30726.0, 30801.0, 30802.0, 30803.0, 30901.0, 30902.0, 30903.0, 30904.0, 30905.0, 30906.0, 30907.0, 30908.0, 31001.0, 31002.0, 31003.0, 31004.0, 31005.0, 31006.0, 31007.0, 31008.0, 31009.0, 31010.0, 31011.0, 31012.0, 31013.0, 31014.0, 31015.0, 31201.0, 31202.0, 31203.0, 31301.0, 31302.0, 31303.0, 31304.0, 31305.0, 31306.0, 31307.0, 31308.0, 31309.0, 31310.0, 31311.0, 31312.0, 31313.0, 31314.0, 31315.0, 31316.0, 31318.0, 31319.0, 31320.0, 31321.0, 31322.0, 31401.0, 31402.0, 31403.0, 31404.0, 31405.0, 40101.0, 40102.0, 40103.0, 40104.0, 40105.0, 40106.0, 40107.0, 40108.0, 40109.0, 40110.0, 40111.0, 40112.0, 40113.0, 40114.0, 40115.0, 40116.0, 40117.0, 40118.0, 40119.0, 40120.0, 40121.0, 40122.0, 40123.0, 40124.0, 40125.0, 40126.0, 40127.0, 40128.0, 40129.0, 40130.0, 40131.0, 40132.0, 40133.0, 40134.0, 40135.0, 40136.0, 40137.0, 40138.0, 40139.0, 40140.0, 40141.0, 40142.0, 40143.0, 40144.0, 40145.0, 40146.0, 40147.0, 40148.0, 40149.0, 40150.0, 40151.0, 40152.0, 40153.0, 40154.0, 40155.0, 40156.0, 40157.0, 40158.0, 40159.0, 40160.0, 40161.0, 40162.0, 40163.0, 40164.0, 40165.0, 40166.0, 40167.0, 40168.0, 40169.0, 40170.0, 40171.0, 40172.0, 40201.0, 40202.0, 40203.0, 40204.0, 40205.0, 40206.0, 40207.0, 40208.0, 40209.0, 40210.0, 40211.0, 40212.0, 40301.0, 40302.0, 40303.0, 40304.0, 40401.0, 40402.0, 40403.0, 40501.0, 40502.0, 40701.0, 40901.0, 41001.0, 41301.0, 41302.0, 41303.0, 41304.0, 41401.0, 41402.0, 41403.0, 41404.0, 41405.0, 41406.0, 50201.0, 50302.0, 50303.0, 50401.0, 50402.0, 50403.0, 50404.0, 50405.0, 50406.0, 50407.0, 50408.0, 50409.0, 50410.0, 50411.0, 50412.0, 50413.0, 50414.0, 50415.0, 50416.0, 50417.0, 50501.0, 50502.0, 50503.0, 50504.0, 50505.0, 50506.0, 50507.0, 50508.0, 50509.0, 50510.0, 50511.0, 50512.0, 50513.0, 50514.0, 50515.0, 50516.0, 50517.0, 50518.0, 50519.0, 50520.0, 50521.0, 50522.0, 50523.0, 50524.0, 50525.0, 50526.0, 50527.0, 50528.0, 50529.0, 50530.0, 50601.0, 50602.0, 50603.0, 50604.0, 50605.0, 50606.0, 50607.0, 50608.0, 50609.0, 50610.0, 50611.0, 50612.0, 50613.0, 50614.0, 50615.0, 50616.0, 50617.0, 50701.0, 50801.0, 50802.0, 50803.0, 50804.0, 50805.0, 50806.0, 50807.0, 50808.0, 50809.0, 50810.0, 50811.0, 50812.0, 50813.0, 50814.0, 50815.0, 51101.0, 51102.0, 51103.0, 51104.0, 51105.0, 51106.0, 51107.0, 51108.0, 51109.0, 51110.0, 51111.0, 51112.0, 51113.0, 51114.0, 51115.0, 51116.0, 51117.0, 51118.0, 51201.0, 51202.0, 51203.0, 51301.0, 51302.0, 51303.0, 51304.0, 51305.0, 51306.0, 51307.0, 51308.0, 51309.0, 51310.0, 51311.0, 51312.0, 51313.0, 60101.0, 60102.0, 60103.0, 60104.0, 60105.0, 60106.0, 60107.0, 60108.0, 60109.0, 60111.0, 60112.0, 60113.0, 60114.0, 60115.0, 60116.0, 60117.0, 60118.0, 60119.0, 60120.0, 60121.0, 60122.0, 60124.0, 60125.0, 60126.0, 60127.0, 60128.0, 60129.0, 60130.0, 60131.0, 60132.0, 60133.0, 60134.0, 60135.0, 60136.0, 60137.0, 60138.0, 60139.0, 60140.0, 60141.0, 60142.0, 60143.0, 60144.0, 60145.0, 60146.0, 60147.0, 60148.0, 60149.0, 60150.0, 60151.0, 60152.0, 60153.0, 60154.0, 60155.0, 60156.0, 60157.0, 60158.0, 60159.0, 60160.0, 60161.0, 60162.0, 60163.0, 60164.0, 60165.0, 60166.0, 60167.0, 60168.0, 60169.0, 60170.0, 60171.0, 60173.0, 60174.0, 60175.0, 60176.0, 60177.0, 60178.0, 60179.0, 60180.0, 60181.0, 60182.0, 60201.0, 60202.0, 60204.0, 60205.0, 60206.0, 60207.0, 60209.0, 60210.0, 60211.0, 60212.0, 60213.0, 60214.0, 60215.0, 60216.0, 60217.0, 60218.0, 60219.0, 60220.0, 60221.0, 60222.0, 60223.0, 60224.0, 60225.0, 60226.0, 60227.0, 60228.0, 60229.0, 60230.0, 60232.0, 60233.0, 60235.0, 60301.0, 60302.0, 60303.0, 60304.0, 60305.0, 60306.0, 60307.0, 60308.0, 60309.0, 60310.0, 60401.0, 60402.0, 60404.0, 60702.0, 60703.0, 60704.0, 60707.0, 60708.0, 60709.0, 60710.0, 60801.0, 60802.0, 60803.0, 60805.0, 60902.0, 60903.0, 60904.0, 60905.0, 60906.0, 60907.0, 60908.0, 60909.0, 61001.0, 61002.0, 61003.0, 61004.0, 61005.0, 61006.0, 61007.0, 61008.0, 61010.0, 61201.0, 61301.0, 61303.0, 61304.0, 61305.0, 61306.0, 61307.0, 61308.0, 61309.0, 61312.0, 61313.0, 61314.0, 61315.0, 61316.0, 61401.0, 61402.0, 61403.0, 61404.0, 61405.0, 61406.0, 61407.0, 70101.0, 70102.0, 70103.0, 70104.0, 70105.0, 70106.0, 70107.0, 70108.0, 70109.0, 70110.0, 70111.0, 70112.0, 70113.0, 70114.0, 70115.0, 70116.0, 70117.0, 70201.0, 70202.0, 70203.0, 70204.0, 70701.0, 70702.0, 80101.0, 80102.0, 80401.0, 80402.0, 80403.0, 80404.0, 80405.0, 80406.0, 80407.0, 80408.0, 80409.0, 80410.0, 80411.0, 80412.0, 80413.0, 80414.0, 80415.0, 80416.0, 80417.0, 80418.0, 80419.0, 80420.0, 80421.0, 80422.0, 80423.0, 80424.0, 80425.0, 80426.0, 80427.0, 80428.0, 80429.0, 80430.0, 80431.0, 80432.0, 80433.0, 80434.0, 80435.0, 80436.0, 80437.0, 80438.0, 80439.0, 80440.0, 80441.0, 80442.0, 80443.0, 80444.0, 80445.0, 80446.0, 80501.0, 80502.0, 80503.0, 80504.0, 80505.0, 80506.0, 80507.0, 80508.0, 80509.0, 80510.0, 80511.0, 80512.0, 80513.0, 80514.0, 80515.0, 80516.0, 80517.0, 80518.0, 80519.0, 80520.0, 80521.0, 80601.0, 80602.0, 80603.0, 80604.0, 80605.0, 80606.0, 80607.0, 80608.0, 80609.0, 80610.0, 80611.0, 80801.0, 80802.0, 80803.0, 80804.0, 80805.0, 80806.0, 80807.0, 80808.0, 80809.0, 80810.0, 80811.0, 80812.0, 80813.0, 80814.0, 80815.0, 80816.0, 80817.0, 80818.0, 80901.0, 80902.0, 80903.0, 80904.0, 80905.0, 80906.0, 80907.0, 80908.0, 81001.0, 81002.0, 81003.0, 81004.0, 81005.0, 81006.0, 81007.0, 81008.0, 81009.0, 81010.0, 81011.0, 81012.0, 81013.0, 81014.0, 81015.0, 81016.0, 81017.0, 81018.0, 81019.0, 81020.0, 81021.0, 81022.0, 81101.0, 81102.0, 81103.0, 81104.0, 81105.0, 81106.0, 81107.0, 81108.0, 81109.0, 81110.0, 81111.0, 81112.0, 81113.0, 81114.0, 81201.0, 81202.0, 81203.0, 81204.0, 81205.0, 81206.0, 81207.0, 81208.0, 81209.0, 81210.0, 81211.0, 81212.0, 81213.0, 81214.0, 81215.0, 81216.0, 81217.0, 81218.0, 81219.0, 81220.0, 81221.0, 81222.0, 81301.0, 81302.0, 81303.0, 81304.0, 81305.0, 81306.0, 81307.0, 81308.0, 81309.0, 81310.0, 81311.0, 81312.0, 81313.0, 81314.0, 81315.0, 81316.0, 81317.0, 81318.0, 81319.0, 81320.0, 81321.0, 81322.0, 81323.0, 81324.0, 81325.0, 81326.0, 81327.0, 81328.0, 81329.0, 81330.0, 81331.0, 81332.0, 81333.0]
                     },
    'HFP2009_WGS84': {'min_val': 0.0, 'max_val': 50.0, 'nan': -9999, 'new_nan': -1, 'mu': 5.855772, 'sigma': 6.8387604, 'res': 35.613844445368, 'cat':None},
    'HFP1993_WGS84': {'min_val': 0.0, 'max_val': 50.0, 'nan': -9999, 'new_nan': -1, 'mu': 5.406442, 'sigma': 6.529244, 'res': 35.613844445368, 'cat':None},
    'wc2.1_30s_bio_1':{'min_val': -54.770832, 'max_val': 31.3875, 'nan': -3.3999999521443642e+38, 'new_nan': -56, 'mu': -4.4446025, 'sigma': 24.706017, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_10':{'min_val': -38.716667, 'max_val': 38.533333, 'nan': -3.3999999521443642e+38, 'new_nan': -40, 'mu': 6.941778, 'sigma': 21.19865, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_11':{'min_val': -66.416664, 'max_val': 29.3, 'nan': -3.3999999521443642e+38, 'new_nan': -68, 'mu': -14.421264, 'sigma': 26.796232, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_12':{'min_val': 0.0, 'max_val': 11256.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 532.14465, 'sigma': 630.03644, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_13':{'min_val': 0.0, 'max_val': 2982.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 91.45205, 'sigma': 102.122856, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_14':{'min_val': 0.0, 'max_val': 528.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 14.405708, 'sigma': 27.11674, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_15':{'min_val': 0.0, 'max_val': 235.28041, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 75.82189, 'sigma': 44.050907, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_16':{'min_val': 0.0, 'max_val': 6637.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 235.6551, 'sigma': 274.2746, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_17':{'min_val': 0.0, 'max_val': 1649.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 52.30186, 'sigma': 90.52435, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_18':{'min_val': 0.0, 'max_val': 6124.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 152.55205, 'sigma': 183.28079, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_19':{'min_val': 0.0, 'max_val': 5645.0, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 104.11981, 'sigma': 181.23813, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_2':{'min_val': 1.0, 'max_val': 22.233334, 'nan': -3.3999999521443642e+38, 'new_nan': -0, 'mu': 10.0962305, 'sigma': 3.1111703, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_3':{'min_val': 8.243729, 'max_val': 100.0, 'nan': -3.3999999521443642e+38, 'new_nan': 7, 'mu': 34.370358, 'sigma': 18.710659, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_4':{'min_val': 0.0, 'max_val': 2382.5247, 'nan': -3.3999999521443642e+38, 'new_nan': -1, 'mu': 891.212, 'sigma': 467.2958, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_5':{'min_val': -31.2, 'max_val': 48.6, 'nan': -3.3999999521443642e+38, 'new_nan': -33, 'mu': 13.787962, 'sigma': 21.70779, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_6':{'min_val': -72.6, 'max_val': 26.5, 'nan': -3.3999999521443642e+38, 'new_nan': -74, 'mu': -20.40257, 'sigma': 26.016174, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_7':{'min_val': 1.0, 'max_val': 72.8, 'nan': -3.3999999521443642e+38, 'new_nan': 0.0, 'mu': 34.190613, 'sigma': 12.002961, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_8':{'min_val': -66.35, 'max_val': 38.016666, 'nan': -3.3999999521443642e+38, 'new_nan': -68, 'mu': -1.3610482, 'sigma': 29.150635, 'res': 30, 'cat':None},
    'wc2.1_30s_bio_9':{'min_val': -57.183334, 'max_val': 37.683334, 'nan': -3.3999999521443642e+38, 'new_nan': -59, 'mu': -5.745067, 'sigma': 22.773882, 'res': 30, 'cat':None},
    'bdod_0-5cm_mean_1000_WGS84':{'min_val': 7.0, 'max_val': 182.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 111.14117, 'sigma': 29.322037, 'res': 32.77676792018758, 'cat':None},
    'cec_0-5cm_mean_1000_WGS84':{'min_val': 10.0, 'max_val': 1043.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 283.2858, 'sigma': 138.39671, 'res': 32.77676792018758, 'cat':None},
    'cfvo_0-5cm_mean_1000_WGS84':{'min_val': 0.0, 'max_val': 808.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 126.41906, 'sigma': 68.653915, 'res': 32.77676792018758, 'cat':None},
    'clay_0-5cm_mean_1000_WGS84':{'min_val': 8.0, 'max_val': 796.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 236.73318, 'sigma': 76.499245, 'res': 32.77676792018758, 'cat':None},
    'nitrogen_0-5cm_mean_1000_WGS84':{'min_val': 15.0, 'max_val': 30992.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 4641.656, 'sigma': 3697.1582, 'res': 32.77676792018758, 'cat':None},
    'ocd_0-5cm_mean_1000_WGS84':{'min_val': 35.0, 'max_val': 1199.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 393.91708, 'sigma': 193.66812, 'res': 32.77676792018758, 'cat':None},
    'ocs_0-30cm_mean_1000_WGS84':{'min_val': 4.0, 'max_val': 265.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 51.37108, 'sigma': 23.89384, 'res': 32.77676792018758, 'cat':None},
    'phh2o_0-5cm_mean_1000_WGS84':{'min_val': 34.0, 'max_val': 103.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 63.444016, 'sigma': 11.182072, 'res': 32.77676792018758, 'cat':None},
    'sand_0-5cm_mean_1000_WGS84':{'min_val': 4.0, 'max_val': 982.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 446.11685, 'sigma': 135.35095, 'res': 32.77676792018758, 'cat':None},
    'silt_0-5cm_mean_1000_WGS84':{'min_val': 3.0, 'max_val': 894.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 317.1515, 'sigma': 106.88893, 'res': 32.77676792018758, 'cat':None},
    'soc_0-5cm_mean_1000_WGS84':{'min_val': 8.0, 'max_val': 4922.0, 'nan': -32768.0, 'new_nan': -1, 'mu': 688.285, 'sigma': 687.8051, 'res': 32.77676792018758, 'cat':None}
}


def Standardize(raster):
    if raster.cat is None:
        raster.raster = np.where(raster.raster != raster.new_nan, (raster.raster-raster.mu)/raster.sigma, raster.new_nan)


def Normalize(raster):
    if raster.cat is None:
        if raster.log_transform:
            raster.raster = np.where(raster.raster != raster.new_nan, np.log(raster.raster + 1 - raster.min_val), -1)
        else:
            raster.raster = np.where(raster.raster != raster.new_nan, (raster.raster - raster.min_val) / (raster.max_val - raster.min_val), -1)
        raster.new_nan = -1



class Raster(object):
    """
    Raster is dedicated to a single raster management...
    """
    def __init__(self, path, name, metadata, epsg_ref=4326, transform=None, log_transform=False):
        """
        Loads a tiff file describing an environmental raster into a numpy array and...

        :type new_nan:
        :param path: the path of the raster (the directory)
        :param nan: the value to use when NaN number are present. If False, then default values will be used
        :param normalized: if True the raster will be normalized (minus the mean and divided by std)
        :param epsg_ref: the reference coordinates system to access the raster with lon/lat decimal degrees coordinates
        :param transform: if a function is given, it will be applied on each patch.
        :param size: the size of a patch (size x size)
        """
        # print("metadata.keys:", metadata.keys())
        self.path = path
        self.name = name
        self.nan = metadata['nan']
        self.crs_ref = CRS.from_epsg(epsg_ref)
        self.new_nan = metadata['new_nan']
        self.mu = metadata['mu']
        self.sigma = metadata['sigma']
        self.min_val = metadata['min_val']
        self.max_val = metadata['max_val']
        self.raster_res = metadata['res']
        self.cat = metadata['cat']
        self.transform = transform
        self.log_transform = log_transform
        self.sampled4plot = "/gpfsscratch/rech/pcz/uzk84jj/sampled4plot/"

        if self.name=='ecoregions001':
            # Dict to assign new categories values
            keys   = np.sort(self.cat).astype(int)
            values = np.arange(1, len(keys) + 1)
            self.D               = dict(zip(keys, values))
            self.D[self.new_nan] = self.new_nan
            # 14 possible biomes
            self.biomes_ref      = set(sorted([int(str(l)[1:3]) for l in self.D.keys()]))

        path = re.sub(r'/\/+/', '/', path)
        # to avoid the annoying corresponding warning, temporary warning disabling...
        warnings.filterwarnings("ignore")
        src = rasterio.open(path + self.name + '.tif', nodata=self.nan)

        warnings.filterwarnings("default")

        # print("src.meta['crs']:", src.meta['crs'])
        if src.meta['crs'] is None:

            with open(path + '/' + 'GeoMetaData.csv') as f:
                metadata = f.read()

            m_split = metadata.split('\n')[1].split(';')
            # loading file data
            self.crs = self.crs_ref
            self.x_min = float(m_split[1])
            self.y_min = float(m_split[2])
            self.x_resolution = float(m_split[5])
            self.y_resolution = float(m_split[6])
            self.n_rows = int(m_split[3])
            self.n_cols = int(m_split[4])
        else:
            self.crs   = CRS(src.crs)
            self.x_min = src.bounds.left
            self.y_min = src.bounds.bottom
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width
        # print(self.x_min, self.y_min, self.x_resolution, self.y_resolution, self.n_rows, self.n_cols)
        # some tiff do not contain geo data (stored in the file GeoMetaData)
        # loading the raster
        self.raster = np.squeeze(src.read())
        src.close()

        # value bellow min_value are considered incorrect and therefore no_data
        # print("***", self.name, "***")
        # print("self.nan: *"+str(self.nan)+"*")
        self.raster[np.isnan(self.raster)] = self.new_nan
        self.raster = np.where(self.raster == self.nan, self.new_nan, self.raster)
        # self.raster[self.raster == self.nan] = self.new_nan
        # print("self.raster[self.raster == self.nan]    :", self.raster[self.raster == self.nan])
        # print("self.raster[self.raster == self.new_nan]:", self.raster[self.raster == self.new_nan])
        # print("\n")

        # Rasters with multiple negative nan values
        if self.name in ["Roads_WGS84"]:
            self.raster[self.raster < self.nan] = self.new_nan

        if self.transform:
            for t in self.transform:
                t(self)

        # deals with the coordinates projection needed when the raster is not in WGS84 (=epsg_ref)
        if self.crs!=self.crs_ref:
            print(' (' + self.name + ' is not in WGS84 projection. Use the script data/util/reproject_raster.py to project the raster in WGS84 first.)')

        # setting the shape of the raster
        self.shape = self.raster.shape

    def get_patch(self, lat, lng, size, res):
        """
        Avoid using this method directly
        :return: a patch
        """
        # conversion arc/second to meters
        data_pixel_size_lat = EQUATOR_ARC_SECOND_IN_METERS  # constant for lat
        data_pixel_size_lng = EQUATOR_ARC_SECOND_IN_METERS * math.cos(math.radians(lat))  # depends on lat for lng
        # for the extraction of one patch, all data are considered from this resolution

        #  try:
        #  equivalent to .index() method, but .index() needs src to be still opened!
        row_num = int(self.n_rows - (lat - self.y_min) / self.y_resolution)
        col_num = int((lng - self.x_min) / self.x_resolution)


        # environmental vector
        if size == 1:
            patch = self.raster[row_num, col_num].astype(np.float)

        else:
            brut_patch_size = (max(4,np.ceil((size * 1.5) * (res / (data_pixel_size_lat * self.raster_res)))),
                               max(4,np.ceil((size * 1.5) * (res / (data_pixel_size_lng * self.raster_res))))
                               )
            # print("brut_patch_size:", brut_patch_size)
            half_size = int(brut_patch_size[0]/2), int(brut_patch_size[1]/2)
            brut_patch = self.raster[
                    row_num-half_size[0]:row_num+half_size[0],
                    col_num - half_size[1]:col_num+half_size[1]
                    ].astype(np.float)
            # print("brut_patch.shape:", brut_patch.shape)
            corrected_data_running_shape = (round(brut_patch_size[0] * ((data_pixel_size_lat * self.raster_res) / res)), # why  * brut_patch_size?
                                            round(brut_patch_size[1] * ((data_pixel_size_lng * self.raster_res) / res)))
            # print("corrected_data_running_shape:", corrected_data_running_shape)
            # print("np.stack((brut_patch, brut_patch, brut_patch), axis=2).shape:", np.stack((brut_patch, brut_patch, brut_patch), axis=2).shape)
            interpolation = cv2.INTER_LINEAR if self.cat is None else cv2.INTER_NEAREST
            try:
                patch = cv2.resize(np.stack((brut_patch, brut_patch, brut_patch), axis=2),
                                   dsize=(corrected_data_running_shape[1], corrected_data_running_shape[0]),
                                   interpolation=interpolation)
            except cv2.error:
                print(str(cv2.error)+": Occurrence at lat/lon: "+str(lat)+"//"+str(lng)+" probably outside raster "+self.name+" boundaries.")
                patch = np.float64(self.new_nan) if size == 1 else np.full((size, size), self.new_nan)
            else:
                patch = patch[:, :, 2]


            # true center is in bottom right of the center pixels if the size is an odd number         OK
            y, x = patch.shape
            startx = x // 2 - (size // 2)
            starty = y // 2 - (size // 2)

            patch = patch[starty:starty + size, startx:startx + size]
        """
        except Exception as err:
            print(err)
            print("error extracting occ. at lat/lon", lat, lng, "in raster", self.name)
            exit()
            if size == 1:
                patch = np.float64(self.new_nan)
            else:
                patch = np.full((size, size), self.new_nan)
        """
        # Saves patches before cat rasters stacking
        patch_path = self.sampled4plot + '_'.join(["lat", str(lat), "lon", str(lng)]) + "/"
        Path(patch_path).mkdir(parents=True, exist_ok=True)
        patch_name = '_'.join(["lat", str(lat), "lon", str(lng), self.name, "cat", str(self.cat is not None)])
        np.save(patch_path+patch_name, patch, allow_pickle=True)


        # patch = patch[np.newaxis] if self.cat is None else self.compute_cat_patch(patch)
        patch = patch[np.newaxis] if self.cat is None else (self.compute_ecoregions_per_biome_patch(patch) if self.name=='ecoregions001' else self.compute_cat_patch(patch))

        return patch

    def compute_cat_patch(self, patch):
        # Categorial raster
        # print("***", self.name, "***")
        patch = np.stack([np.where(patch != self.new_nan, patch == c, self.new_nan) for c in self.cat], axis=0)
        return patch

    def compute_ecoregions_per_biome_patch(self, patch):
        # print("***", self.name, "***")
        patch = patch.astype(int)
        # Patch with new values from dictionary self.D
        indexer  = np.array([self.D.get(i, -1) for i in range(patch.min(), patch.max() + 1)])
        patch_nv = indexer[(patch - patch.min())]

        S          = np.stack([np.where(patch != -1, patch // 100 % 100 == b, -1) for b in self.biomes_ref])
        indices    = np.where(S == 1)
        S[indices] = patch_nv[indices[1:]]
        # print("S.shape   :", S.shape)
        # print("Before:\n")
        # print("S[indices]:", S[indices])
        # print("S[:,:2,:2]:", S[:,:2,:2])
        # To dispatch original ecoregions codes in S instead of new ones, simply replace patch_nv by patch
        if self.transform is not None and Normalize in self.transform:
            S = S / len(self.D)
        return S

    def __len__(self):
        """
        :return: the depth of the tensor/vector...
        """
        return 1


class RasterProvider(object):
    """
    PatchExtractor enables the extraction of an environmental tensor from multiple rasters given a GPS
    position.
    """
    def __init__(self, root_path, raster_list=None, raster_transform=None, patch_transform=None, log_transform=False, size=256, res=100):
        self.root_path = root_path
        self.raster_list = raster_list
        self.raster_transform = raster_transform
        self.patch_transform = patch_transform
        self.log_transform = log_transform
        self.size = size
        self.res = res
        self.rasters = []

        if not self.raster_list:
            self.add_all()
        else:
            for raster in self.raster_list:
                self.append(raster)

    def add_all(self):
        """
        Add all variables (rasters) available at root_path
        """
        for key in sorted(raster_metadata.keys()):
            if 'ignore' not in raster_metadata[key]:
                self.append(key)

    def append(self, raster_name):
        """
        This method append a new raster given its name

        :param raster_name:
        """
        # you may want to add rasters one by one if specific configuration are required on a per raster
        # basis
        print('Adding: ' + raster_name)
        try:
            r = Raster(self.root_path, raster_name, raster_metadata[raster_name], transform=self.raster_transform, log_transform=self.log_transform)
            self.rasters.append(r)
        except rasterio.errors.RasterioIOError:
            print(' (' + raster_name + ' not available...)')

    def clean(self):
        """
        Remove all rasters from the extractor.
        """
        print('Removing all rasters...')
        self.rasters = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_ = ''

        def raster_str(r):
            result = ''
            result += '-' * 50 + '\n'
            result += 'title: ' + r.name + '\n'
            result += '\t x_min: ' + str(r.x_min) + '\n'
            result += '\t y_min: ' + str(r.y_min) + '\n'
            result += '\t x_resolution: ' + str(r.x_resolution) + '\n'
            result += '\t y_resolution: ' + str(r.y_resolution) + '\n'
            result += '\t n_rows: ' + str(r.n_rows) + '\n'
            result += '\t n_cols: ' + str(r.n_cols) + '\n'
            return result
        for r in self.rasters:
            str_ += raster_str(r)

        return str_

    def get_rasters_order(self):
        return [r.name for r in self.rasters]

    def __getitem__(self, item):
        """
        :param item: the id and GPS location (id, latitude, longitude)
        :return: return the environmental tensor or vector (size>1 or size=1)
        """
        if len(self.rasters) > 1:
            patch = np.concatenate([r.get_patch(item[1], item[2], size=self.size, res=self.res) for r in self.rasters])
        else:
            patch = self.rasters[0].get_patch(item[1], item[2], size=self.size, res=self.res)
        if self.patch_transform:
            for transform in self.patch_transform:
                patch = transform(patch)
        return patch

    def __len__(self):
        """
        :return: the number of variables (not the size of the tensor when some variables have a one hot encoding
                 representation)
        """
        return len(self.rasters)
