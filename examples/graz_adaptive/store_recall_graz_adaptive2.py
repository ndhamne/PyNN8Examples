import spynnaker8 as p
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

p.setup(1)
runtime=2500

#### Create Populations ####
# create input neurons
spike_times = [
[514, 522, 559, 562, 578, 1006, 1023, 1107, 1112, 1157, 1166, 1177, 1182, 1194, 1226, 1236, 1240, 1247, 1251, 1283, 1301, 1315, 1368, 1383, 1600, 1626, 1646, 1655, 1713, 1733, 1771, 1773, 1785, 1797, 1809, 1835, 1856, 1864, 1895, 1933, 1945, 1959, 1962, 1970, 1991, 2007, 2008, 2016, 2027, 2029, 2031, 2061, 2121, 2174, 2185, 2190, 2194, 2198],
[402, 412, 418, 444, 450, 460, 472, 543, 1006, 1083, 1111, 1117, 1119, 1158, 1178, 1193, 1207, 1227, 1253, 1277, 1287, 1293, 1311, 1319, 1323, 1330, 1369, 1654, 1657, 1690, 1692, 1693, 1696, 1698, 1734, 1747, 1759, 1760, 1788, 1790, 1825, 1869, 1904, 1907, 1933, 1935, 1936, 1979, 1992, 1996, 2003, 2031, 2046, 2101, 2106, 2132, 2141, 2146, 2147, 2183, 2186],
[413, 419, 457, 464, 484, 529, 1031, 1048, 1063, 1120, 1126, 1174, 1185, 1188, 1230, 1237, 1262, 1266, 1284, 1289, 1295, 1344, 1358, 1379, 1613, 1630, 1632, 1673, 1692, 1706, 1718, 1733, 1735, 1755, 1783, 1805, 1818, 1828, 1867, 1884, 1913, 1926, 1953, 1997, 2005, 2065, 2067, 2096, 2101, 2102, 2130, 2142, 2149, 2161, 2182],
[401, 413, 426, 448, 491, 504, 511, 516, 543, 566, 576, 1014, 1076, 1081, 1098, 1110, 1135, 1151, 1195, 1196, 1214, 1215, 1243, 1286, 1302, 1311, 1318, 1322, 1332, 1389, 1395, 1645, 1665, 1671, 1676, 1700, 1748, 1755, 1762, 1767, 1833, 1836, 1850, 1852, 1878, 1884, 1886, 1898, 1899, 1934, 1936, 1937, 1957, 1958, 1961, 2006, 2030, 2031, 2036, 2053, 2079, 2081, 2086, 2099, 2106, 2122, 2131, 2152, 2154, 2165, 2192, 2198],
[417, 436, 441, 446, 447, 452, 455, 466, 516, 543, 553, 555, 560, 586, 588, 593, 1035, 1060, 1074, 1101, 1143, 1150, 1157, 1168, 1178, 1194, 1211, 1243, 1259, 1270, 1289, 1299, 1302, 1318, 1325, 1338, 1617, 1623, 1636, 1639, 1764, 1816, 1830, 1848, 1922, 1924, 1942, 1945, 1954, 2028, 2030, 2037, 2050, 2088, 2108, 2131, 2145, 2151, 2156, 2166, 2182, 2194, 2197],
[413, 476, 523, 526, 529, 539, 582, 1027, 1060, 1102, 1109, 1121, 1182, 1187, 1214, 1228, 1229, 1237, 1243, 1293, 1327, 1360, 1669, 1700, 1822, 1841, 1861, 1888, 1889, 1894, 1900, 1901, 1958, 1973, 1984, 2002, 2025, 2030, 2045, 2081, 2119, 2149, 2156, 2164, 2183, 2187],
[403, 406, 413, 431, 439, 450, 474, 484, 500, 538, 562, 564, 575, 1003, 1006, 1031, 1053, 1094, 1104, 1115, 1125, 1126, 1145, 1165, 1185, 1202, 1224, 1259, 1317, 1323, 1338, 1341, 1344, 1377, 1382, 1387, 1600, 1609, 1646, 1676, 1741, 1748, 1752, 1774, 1784, 1791, 1795, 1811, 1813, 1819, 1842, 1854, 1856, 1872, 1874, 1880, 1881, 1910, 1978, 2039, 2068, 2110, 2140, 2179],
[405, 412, 445, 446, 458, 469, 491, 513, 546, 562, 573, 579, 590, 591, 1163, 1166, 1167, 1169, 1175, 1190, 1196, 1199, 1225, 1253, 1270, 1302, 1313, 1322, 1342, 1391, 1397, 1624, 1637, 1638, 1653, 1654, 1669, 1709, 1722, 1734, 1761, 1775, 1793, 1807, 1836, 1840, 1849, 1869, 1933, 1964, 1978, 1993, 2008, 2012, 2014, 2051, 2054, 2064, 2065, 2086, 2118, 2142, 2165, 2179, 2181, 2197],
[420, 429, 431, 444, 455, 457, 512, 516, 536, 562, 592, 594, 1000, 1017, 1028, 1069, 1103, 1120, 1124, 1132, 1139, 1181, 1223, 1287, 1308, 1310, 1317, 1326, 1350, 1385, 1388, 1390, 1646, 1677, 1702, 1729, 1739, 1758, 1761, 1764, 1766, 1775, 1790, 1799, 1800, 1812, 1827, 1836, 1883, 1893, 1914, 1922, 1927, 1953, 1976, 1997, 2028, 2054, 2062, 2071, 2073, 2093, 2095, 2126, 2151, 2159, 2165, 2170],
[442, 452, 458, 465, 487, 557, 1012, 1028, 1030, 1042, 1090, 1138, 1216, 1271, 1291, 1307, 1314, 1322, 1327, 1354, 1366, 1611, 1612, 1623, 1675, 1680, 1703, 1726, 1759, 1763, 1791, 1803, 1815, 1824, 1848, 1915, 1969, 1972, 1992, 2008, 2033, 2035, 2037, 2057, 2108, 2109, 2114, 2128, 2132, 2141, 2156, 2188],
[424, 428, 434, 447, 459, 462, 483, 494, 511, 533, 575, 579, 1014, 1018, 1020, 1150, 1172, 1175, 1269, 1270, 1276, 1292, 1300, 1302, 1317, 1322, 1360, 1374, 1381, 1394, 1604, 1630, 1641, 1648, 1694, 1699, 1701, 1711, 1722, 1780, 1784, 1799, 1836, 1858, 1948, 1965, 2010, 2033, 2042, 2047, 2056, 2065, 2069, 2092, 2098, 2145, 2179, 2199],
[401, 403, 408, 420, 427, 460, 471, 479, 499, 525, 537, 540, 547, 562, 576, 1008, 1019, 1025, 1029, 1038, 1042, 1058, 1116, 1192, 1239, 1248, 1275, 1326, 1329, 1364, 1379, 1384, 1622, 1631, 1640, 1693, 1760, 1764, 1791, 1851, 1852, 1866, 1912, 2067, 2068, 2069, 2077, 2100, 2137, 2150, 2157, 2184],
[410, 446, 448, 510, 512, 526, 556, 562, 572, 584, 1016, 1095, 1121, 1130, 1159, 1162, 1169, 1194, 1220, 1227, 1239, 1247, 1273, 1286, 1291, 1296, 1298, 1306, 1308, 1311, 1322, 1343, 1345, 1382, 1391, 1648, 1667, 1721, 1754, 1765, 1791, 1794, 1829, 1858, 1866, 1880, 1938, 1942, 1943, 1960, 1986, 2002, 2018, 2042, 2059, 2061, 2070, 2077, 2083, 2110, 2116, 2150, 2154, 2164],
[410, 425, 450, 451, 461, 487, 495, 499, 500, 527, 536, 540, 556, 585, 586, 594, 598, 1005, 1022, 1027, 1028, 1034, 1041, 1043, 1057, 1143, 1151, 1172, 1174, 1245, 1264, 1269, 1282, 1341, 1377, 1379, 1665, 1685, 1700, 1703, 1745, 1764, 1782, 1806, 1818, 1844, 1876, 1882, 1893, 1905, 1956, 1963, 1995, 2002, 2026, 2049, 2061, 2081, 2101, 2127, 2137, 2170, 2192, 2194, 2196],
[421, 439, 471, 481, 487, 491, 515, 524, 538, 551, 1006, 1019, 1034, 1035, 1037, 1041, 1058, 1101, 1102, 1119, 1124, 1130, 1139, 1152, 1191, 1198, 1202, 1212, 1222, 1248, 1264, 1266, 1294, 1301, 1309, 1330, 1332, 1338, 1339, 1370, 1386, 1389, 1391, 1392, 1604, 1617, 1659, 1666, 1697, 1714, 1734, 1742, 1746, 1773, 1818, 1838, 1846, 1871, 1891, 1912, 1927, 1936, 1969, 2002, 2018, 2036, 2061, 2069, 2091, 2113, 2125, 2146],
[411, 491, 502, 598, 1080, 1082, 1086, 1090, 1092, 1115, 1118, 1165, 1167, 1168, 1188, 1189, 1208, 1214, 1221, 1232, 1234, 1271, 1278, 1300, 1320, 1328, 1336, 1345, 1346, 1357, 1367, 1376, 1388, 1632, 1650, 1672, 1698, 1700, 1701, 1765, 1779, 1784, 1789, 1797, 1810, 1816, 1899, 1902, 1905, 1912, 1923, 1931, 1935, 1945, 2043, 2055, 2061, 2071, 2094, 2179, 2182],
[401, 420, 433, 446, 473, 475, 522, 544, 554, 563, 592, 1045, 1083, 1085, 1108, 1121, 1169, 1172, 1218, 1251, 1277, 1282, 1286, 1311, 1349, 1379, 1385, 1397, 1606, 1664, 1666, 1843, 1847, 1882, 1883, 1923, 1949, 1961, 1965, 1968, 1977, 2037, 2137, 2151, 2164, 2174, 2178, 2194],
[400, 447, 477, 484, 559, 579, 1050, 1052, 1057, 1066, 1132, 1135, 1154, 1171, 1189, 1222, 1262, 1265, 1273, 1277, 1286, 1295, 1302, 1312, 1394, 1397, 1398, 1626, 1632, 1671, 1685, 1746, 1747, 1754, 1767, 1773, 1788, 1805, 1822, 1826, 1835, 1839, 1863, 1869, 1886, 1899, 1916, 1919, 1925, 1947, 1981, 2028, 2038, 2041, 2066, 2079, 2080, 2132, 2148, 2180],
[411, 455, 477, 514, 520, 531, 567, 572, 591, 1013, 1030, 1033, 1062, 1088, 1090, 1103, 1105, 1135, 1160, 1181, 1184, 1223, 1255, 1265, 1276, 1307, 1310, 1316, 1333, 1344, 1374, 1385, 1833, 1845, 1853, 1884, 1907, 1934, 1943, 1963, 1964, 1966, 1972, 1980, 1990, 1998, 2044, 2061, 2099, 2155, 2179],
[455, 468, 487, 522, 528, 554, 559, 566, 567, 570, 596, 1023, 1052, 1073, 1083, 1100, 1127, 1178, 1181, 1190, 1200, 1206, 1207, 1227, 1235, 1243, 1270, 1277, 1302, 1360, 1363, 1626, 1650, 1651, 1682, 1693, 1700, 1701, 1704, 1752, 1840, 1874, 1890, 1900, 1923, 1973, 1987, 2004, 2007, 2012, 2035, 2044, 2049, 2069, 2070, 2126, 2129, 2144, 2185, 2195],
[428, 434, 501, 506, 541, 542, 1006, 1036, 1062, 1069, 1072, 1084, 1111, 1121, 1136, 1179, 1225, 1228, 1249, 1265, 1273, 1281, 1290, 1339, 1359, 1386, 1604, 1607, 1624, 1625, 1651, 1655, 1672, 1675, 1695, 1699, 1739, 1767, 1778, 1786, 1790, 1882, 1893, 1920, 1921, 1926, 1932, 1944, 1948, 1949, 1955, 1960, 1993, 2004, 2024, 2042, 2067, 2076, 2101, 2127, 2134, 2144, 2162, 2176],
[450, 488, 492, 494, 507, 511, 520, 522, 581, 1006, 1010, 1030, 1032, 1074, 1112, 1118, 1130, 1159, 1163, 1256, 1276, 1284, 1298, 1384, 1611, 1627, 1648, 1657, 1681, 1719, 1721, 1735, 1802, 1814, 1832, 1842, 1847, 1870, 1913, 1923, 1940, 1944, 1958, 1977, 1991, 2000, 2017, 2026, 2071, 2117, 2129, 2146, 2165, 2179, 2183, 2191, 2197],
[409, 432, 440, 453, 454, 464, 470, 533, 540, 578, 597, 1002, 1030, 1087, 1100, 1121, 1147, 1158, 1174, 1192, 1193, 1196, 1250, 1273, 1281, 1312, 1323, 1334, 1355, 1730, 1741, 1751, 1752, 1777, 1790, 1856, 1888, 1923, 1949, 1954, 2016, 2022, 2026, 2035, 2053, 2072, 2086, 2113, 2120, 2121, 2165, 2172, 2185, 2197],
[401, 407, 424, 447, 465, 501, 510, 517, 544, 585, 1003, 1011, 1022, 1028, 1041, 1073, 1143, 1146, 1173, 1175, 1185, 1196, 1238, 1274, 1292, 1328, 1334, 1384, 1601, 1616, 1641, 1669, 1705, 1814, 1819, 1829, 1929, 1936, 1950, 1970, 2017, 2023, 2056, 2068, 2075, 2079, 2084, 2096, 2104, 2126, 2143, 2146, 2149, 2184],
[408, 434, 437, 445, 471, 476, 487, 507, 514, 535, 545, 576, 599, 1016, 1043, 1057, 1060, 1061, 1070, 1084, 1095, 1123, 1130, 1132, 1169, 1203, 1207, 1222, 1254, 1264, 1268, 1289, 1303, 1306, 1308, 1319, 1344, 1345, 1359, 1362, 1363, 1643, 1689, 1697, 1725, 1732, 1741, 1761, 1789, 1809, 1811, 1812, 1822, 1829, 1842, 1898, 1902, 1913, 1928, 1973, 1984, 2021, 2043, 2044, 2081, 2126, 2143, 2153, 2155, 2156, 2168, 2174],
[50, 68, 97, 104, 116, 117, 132, 154, 185, 210, 230, 232, 247, 270, 274, 297, 301, 367, 369, 377, 382, 385, 391, 605, 610, 661, 710, 731, 734, 772, 1409, 1435, 1446, 1456, 1460, 1486, 1504, 1517, 1551, 1555, 2200, 2206, 2236, 2248, 2258, 2305, 2312, 2379, 2385, 2387],
[37, 38, 54, 58, 102, 113, 139, 145, 207, 219, 222, 224, 226, 272, 297, 302, 344, 347, 361, 362, 374, 393, 604, 631, 650, 654, 655, 684, 690, 736, 742, 746, 775, 778, 799, 1416, 1458, 1512, 1517, 1532, 1540, 1557, 1564, 1584, 1599, 2202, 2210, 2222, 2250, 2269, 2349, 2362],
[75, 76, 96, 107, 117, 164, 166, 184, 252, 253, 264, 319, 324, 368, 373, 614, 627, 649, 676, 679, 684, 701, 728, 772, 1422, 1447, 1480, 1511, 1512, 1524, 1533, 1558, 1584, 2205, 2232, 2243, 2246, 2313, 2344, 2363, 2366],
[9, 28, 94, 105, 111, 139, 147, 171, 194, 195, 207, 238, 243, 260, 263, 270, 323, 328, 329, 330, 335, 337, 344, 374, 380, 641, 660, 667, 696, 700, 715, 728, 763, 776, 796, 1447, 1465, 1477, 1491, 1499, 1518, 1553, 1558, 1566, 1585, 2243, 2245, 2263, 2267, 2272, 2297, 2326, 2353, 2364, 2381, 2387, 2397],
[8, 43, 70, 72, 115, 130, 133, 197, 214, 236, 278, 291, 292, 297, 307, 308, 316, 324, 343, 379, 399, 608, 617, 639, 671, 676, 677, 691, 716, 725, 780, 795, 1413, 1459, 1463, 1501, 1514, 1543, 1554, 1570, 2208, 2217, 2251, 2253, 2304, 2352, 2359, 2375, 2387, 2394],
[80, 122, 141, 142, 155, 160, 174, 216, 220, 225, 335, 337, 370, 381, 390, 642, 665, 679, 701, 739, 740, 745, 1412, 1445, 1478, 1500, 1520, 1537, 1551, 1568, 1581, 1583, 2216, 2264, 2270, 2297, 2331, 2340, 2389, 2391, 2395, 2396],
[42, 49, 78, 79, 87, 91, 127, 143, 146, 148, 175, 234, 255, 256, 290, 307, 328, 362, 368, 374, 383, 388, 639, 663, 669, 717, 733, 740, 742, 747, 765, 769, 786, 797, 1423, 1435, 1453, 1462, 1466, 1467, 1493, 1506, 1507, 1508, 1515, 1525, 1593, 2201, 2206, 2278, 2279, 2304, 2310, 2324, 2337, 2350, 2364, 2387, 2396],
[2, 19, 21, 40, 60, 64, 71, 80, 98, 163, 172, 187, 212, 217, 236, 253, 259, 260, 261, 273, 277, 304, 306, 318, 333, 358, 389, 390, 391, 605, 620, 640, 651, 654, 675, 678, 733, 760, 766, 772, 791, 792, 793, 795, 1434, 1440, 1451, 1519, 1523, 1532, 1534, 1537, 1573, 1584, 1585, 2200, 2202, 2227, 2229, 2298, 2299],
[1, 52, 53, 59, 66, 80, 86, 91, 113, 120, 136, 137, 153, 162, 181, 197, 200, 242, 259, 268, 295, 324, 326, 330, 371, 378, 385, 629, 633, 662, 723, 728, 759, 780, 787, 797, 1408, 1409, 1423, 1471, 1487, 1513, 1532, 1559, 1564, 2227, 2260, 2306, 2312, 2324, 2360, 2362],
[19, 56, 60, 74, 88, 93, 95, 108, 117, 175, 186, 212, 216, 239, 270, 277, 306, 318, 325, 343, 351, 376, 616, 625, 652, 657, 660, 689, 691, 713, 725, 732, 740, 753, 758, 765, 776, 789, 794, 796, 1407, 1425, 1426, 1442, 1452, 1477, 1505, 1541, 1542, 1583, 1585, 1588, 1590, 1593, 1596, 2212, 2244, 2255, 2263, 2267, 2296, 2302, 2330, 2350, 2377, 2382, 2388],
[8, 14, 31, 52, 84, 93, 116, 124, 159, 199, 242, 252, 272, 278, 319, 383, 620, 624, 640, 657, 658, 672, 687, 695, 698, 1407, 1416, 1453, 1464, 1474, 1481, 1565, 1578, 1588, 1596, 2263, 2277, 2299, 2337, 2339, 2388],
[39, 40, 68, 80, 83, 156, 180, 184, 186, 214, 223, 296, 309, 326, 343, 359, 395, 398, 610, 616, 618, 619, 684, 686, 687, 699, 731, 734, 737, 739, 749, 753, 780, 783, 788, 792, 1453, 1455, 1476, 1521, 1569, 1596, 2233, 2239, 2251, 2261, 2271, 2304, 2321, 2332, 2364, 2383],
[0, 12, 27, 32, 46, 47, 55, 56, 82, 109, 161, 163, 181, 211, 243, 248, 282, 299, 312, 313, 328, 351, 352, 357, 362, 380, 386, 387, 398, 609, 643, 645, 655, 677, 685, 687, 703, 715, 722, 759, 766, 779, 787, 1404, 1405, 1420, 1425, 1495, 1499, 1510, 1555, 1576, 2214, 2225, 2254, 2269, 2293, 2294, 2313, 2314, 2331, 2344, 2351, 2387, 2394],
[11, 38, 145, 171, 202, 209, 249, 252, 298, 302, 346, 372, 613, 621, 629, 640, 691, 694, 696, 736, 743, 747, 775, 792, 795, 1427, 1462, 1466, 1489, 1498, 1508, 1513, 1566, 2271, 2287, 2304, 2308, 2339, 2340, 2367, 2369, 2380, 2389],
[23, 39, 53, 60, 77, 79, 83, 104, 138, 165, 171, 181, 223, 236, 273, 303, 320, 328, 338, 339, 386, 398, 605, 637, 645, 694, 703, 704, 716, 726, 1404, 1471, 1503, 1525, 1539, 1550, 1569, 1597, 2221, 2225, 2228, 2248, 2264, 2272, 2300, 2320, 2335, 2381, 2389],
[43, 57, 82, 95, 158, 168, 170, 187, 200, 205, 222, 256, 262, 338, 606, 641, 656, 689, 714, 745, 771, 796, 798, 1413, 1432, 1461, 1476, 1492, 1499, 1512, 1568, 1580, 1588, 2265, 2288, 2296, 2333],
[0, 21, 25, 32, 44, 87, 140, 163, 279, 287, 319, 326, 337, 369, 381, 601, 606, 620, 687, 753, 756, 1412, 1413, 1425, 1431, 1473, 1508, 1522, 1554, 1557, 1582, 2244, 2269, 2274, 2292, 2308, 2326, 2342],
[65, 107, 113, 161, 164, 193, 202, 207, 223, 243, 253, 255, 276, 303, 334, 361, 369, 371, 374, 609, 622, 635, 640, 649, 658, 672, 741, 747, 791, 1426, 1434, 1450, 1465, 1473, 1476, 1498, 1519, 1541, 1569, 1580, 1588, 2200, 2222, 2228, 2230, 2241, 2242, 2247, 2286, 2294, 2300, 2330, 2331, 2341, 2345, 2353, 2357, 2390],
[13, 80, 86, 117, 160, 181, 193, 208, 233, 250, 251, 272, 296, 301, 317, 349, 365, 381, 626, 673, 681, 692, 704, 729, 737, 752, 756, 771, 783, 793, 1444, 1449, 1464, 1516, 1517, 1569, 1594, 1598, 2214, 2219, 2287, 2290, 2296, 2299, 2303, 2349, 2368],
[113, 138, 175, 187, 198, 281, 282, 288, 293, 330, 333, 351, 359, 392, 613, 652, 654, 677, 683, 725, 769, 790, 798, 1431, 1441, 1482, 1512, 1520, 1528, 1532, 1537, 1538, 1548, 1590, 2228, 2250, 2293, 2299, 2339, 2381],
[12, 17, 25, 34, 37, 92, 98, 132, 183, 184, 187, 212, 278, 285, 290, 304, 311, 319, 332, 621, 628, 641, 662, 670, 672, 681, 710, 747, 766, 1403, 1405, 1436, 1440, 1447, 1463, 1498, 1509, 1520, 1526, 1533, 2252, 2286, 2323, 2341, 2350, 2392, 2395],
[16, 17, 33, 54, 63, 73, 74, 96, 111, 116, 119, 128, 182, 194, 216, 217, 228, 230, 232, 240, 255, 258, 276, 314, 323, 338, 342, 367, 378, 379, 383, 625, 654, 664, 674, 695, 715, 729, 750, 794, 1401, 1413, 1442, 1492, 1509, 1524, 1582, 1591, 2224, 2240, 2310, 2322, 2330, 2335, 2355],
[13, 16, 17, 29, 48, 63, 66, 68, 72, 92, 99, 121, 127, 140, 155, 159, 183, 218, 220, 253, 288, 341, 351, 365, 390, 616, 617, 639, 674, 679, 681, 684, 705, 755, 766, 1413, 1414, 1449, 1455, 1466, 1469, 1505, 1509, 1533, 1577, 2227, 2233, 2244, 2274, 2300, 2323, 2324, 2341, 2352, 2368, 2369, 2383, 2392],
[7, 21, 24, 57, 63, 150, 156, 163, 172, 178, 190, 195, 202, 229, 255, 260, 269, 289, 297, 299, 339, 340, 362, 382, 602, 658, 682, 725, 780, 1416, 1429, 1433, 1450, 1456, 1462, 1482, 1502, 1532, 1546, 1599, 2248, 2272, 2273, 2291, 2302, 2314, 2348, 2369, 2372, 2379, 2392],
[25, 32, 67, 132, 137, 186, 189, 214, 216, 275, 279, 293, 299, 313, 325, 392, 396, 672, 688, 721, 736, 751, 763, 1404, 1436, 1447, 1452, 1460, 1468, 1471, 1491, 1510, 1511, 1514, 1521, 1543, 1557, 1561, 2211, 2237, 2251, 2303, 2307, 2344, 2360, 2378, 2388],
[17, 69, 155, 158, 180, 193],
[7, 20, 24, 36, 40, 64, 85, 92, 94, 118, 149, 150, 153, 157, 172, 177, 197, 199],
[8, 55, 78, 91, 92, 114, 124, 128, 129, 173, 190, 192, 199],
[13, 65, 80, 119, 125, 140, 162, 165],
[20, 56, 80, 82, 112, 135, 143, 155, 170, 176, 192],
[13, 23, 31, 37, 58, 63, 101, 102, 124, 193, 198],
[11, 18, 21, 57, 61, 98, 119, 123, 179, 185],
[16, 34, 47, 54, 75, 101, 102, 151, 168, 196],
[3, 12, 15, 34, 35, 75, 82, 93, 105, 164, 183, 185, 192, 193],
[6, 43, 58, 67, 71, 76, 132, 159, 186],
[13, 36, 39, 56, 73, 75, 78, 110, 125, 129, 132, 137, 138, 154, 163, 174],
[6, 17, 35, 36, 37, 54, 99, 103, 109, 114, 119, 122, 136, 140, 156, 184, 187],
[7, 20, 32, 35, 47, 50, 67, 78, 101, 170, 173],
[12, 25, 63, 77, 104, 113, 131, 135, 141, 179, 183, 188, 198],
[30, 40, 66, 92, 138, 142, 147, 157, 158, 194],
[69, 104, 154, 163, 176, 191, 192],
[0, 33, 60, 82, 141, 172],
[3, 29, 35, 111, 126, 151, 187, 197],
[0, 17, 40, 46, 57, 105, 124, 141, 163, 180, 181, 183, 184, 186],
[7, 42, 52, 53, 63, 69, 165, 177, 188],
[1, 42, 71, 123, 129, 141, 167, 193, 198],
[12, 34, 65, 84, 124, 154, 170, 190, 194],
[40, 70, 94, 101, 106, 115, 121, 153],
[12, 45, 49, 81, 96, 116, 136, 161, 185],
[12, 17, 25, 48, 71, 92, 107, 139],
[839, 850, 854, 878, 928, 945],
[816, 828, 835, 876, 885, 894, 945, 964, 990, 996],
[809, 812, 830, 831, 884, 930, 955, 966, 979],
[821, 832, 851, 866, 884, 897, 903, 926, 929, 945, 991],
[803, 843, 845, 880, 884, 887, 906, 921, 959, 989],
[844, 848, 850, 902, 911, 917, 938, 939, 943, 966, 978, 982, 993],
[855, 873, 880, 918, 943, 944, 957, 959, 978],
[843, 851, 864, 924, 958, 961],
[818, 865, 872, 882, 883, 891, 898, 901, 921, 931, 948, 952, 957, 969, 978, 984, 997],
[820, 827, 830, 850, 853, 864, 866, 868, 878, 880, 917, 918, 930, 933, 938, 953],
[815, 851, 855, 892, 909, 933, 944, 997],
[823, 827, 843, 870, 893, 939, 941, 949, 950, 953, 976, 978],
[801, 818, 829, 868, 880, 890, 897, 926, 927, 937, 951, 972, 988, 994],
[801, 849, 871, 902, 913, 927, 940, 990, 998],
[806, 813, 832, 844, 868, 905, 910, 916, 941, 960, 976, 989, 993],
[813, 831, 860, 866, 872, 896, 905, 927, 941, 958, 986],
[812, 836, 843, 866, 937, 977],
[805, 911, 928, 996],
[861, 871, 938, 941, 954, 960, 982, 986, 997, 998],
[802, 843, 849, 854, 870, 918, 923, 931, 977],
[823, 841, 858, 878, 883, 884, 929, 930, 935, 981, 993],
[803, 812, 831, 858, 862, 865, 870, 914, 916, 963, 981],
[808, 863, 892, 926, 935, 950, 984],
[804, 811, 829, 849, 877, 900, 910, 923, 932, 998],
[803, 807, 809, 819, 862, 926, 940, 945, 948, 952]
]

pop_input = p.Population(100, p.SpikeSourceArray,
                        {'spike_times': spike_times}, label="input")

# create hidden neurons
hidden_neuron_params = {
    'tau_m': 20.0,
    'cm': 20, # Updated to suit tau_m of 20 and make membrane resistance 1
    'v_rest': 0.0,
#     "i_offset": 200, # dc current
    'thresh_B': 10.0,
    'thresh_b_0': 10,
    'thresh_tau_a': 1200,
    'thresh_beta': 1.7,
    'tau_refrac':3
    }

pop_hidden = p.Population(2,
            p.extra_models.IFCurrDeltaGrazAdaptive(**hidden_neuron_params),
            label='hidden')


# # create output neurons
# output_neuron_params = {
#     'tau_m': 20.0,
#     'cm': 1.0,
#     'v_rest': -65.0,
#     'v_reset': -65.0,
#     "i_offset":0.8, # dc current
#     'v_thresh': -50.0,
#     'v_thresh_resting': -50,
#     'v_thresh_tau': 700,
#     'v_thresh_adaptation': 2,
#     }
#
# pop_output = p.Population(2, p.IF_curr_exp(**output_neuron_paramters),
#                           label='output')

#### Create synaptic connections ####
IH_conn_list_exc=[[0, 0, 0.002, 7],
[1, 0, 0.002, 4],
[2, 0, 0.002, 2],
[3, 0, 0.002, 4],
[4, 0, 0.002, 4],
[5, 0, 0.002, 1],
[6, 0, 0.002, 3],
[7, 0, 0.002, 9],
[8, 0, 0.002, 5],
[9, 0, 0.002, 8],
[10, 0, 0.002, 6],
[11, 0, 0.002, 3],
[12, 0, 0.002, 1],
[13, 0, 0.002, 2],
[14, 0, 0.002, 3],
[15, 0, 0.002, 6],
[16, 0, 0.002, 5],
[17, 0, 0.002, 7],
[18, 0, 0.002, 5],
[19, 0, 0.002, 5],
[20, 0, 0.002, 5],
[21, 0, 0.002, 1],
[22, 0, 0.002, 9],
[23, 0, 0.002, 2],
[24, 0, 0.002, 3],
[50, 0, 0.1, 7],
[51, 0, 0.1, 6],
[52, 0, 0.1, 6],
[53, 0, 0.1, 3],
[54, 0, 0.1, 8],
[55, 0, 0.1, 7],
[56, 0, 0.1, 9],
[57, 0, 0.1, 1],
[58, 0, 0.1, 9],
[59, 0, 0.1, 1],
[60, 0, 0.1, 6],
[61, 0, 0.1, 9],
[62, 0, 0.1, 5],
[63, 0, 0.1, 8],
[64, 0, 0.1, 1],
[65, 0, 0.1, 7],
[66, 0, 0.1, 1],
[67, 0, 0.1, 1],
[68, 0, 0.1, 5],
[69, 0, 0.1, 3],
[70, 0, 0.1, 5],
[71, 0, 0.1, 6],
[72, 0, 0.1, 8],
[73, 0, 0.1, 2],
[74, 0, 0.1, 2],
[75, 0, 0.1, 8],
[76, 0, 0.1, 8],
[77, 0, 0.1, 9],
[78, 0, 0.1, 8],
[79, 0, 0.1, 2],
[80, 0, 0.1, 9],
[81, 0, 0.1, 1],
[82, 0, 0.1, 6],
[83, 0, 0.1, 2],
[84, 0, 0.1, 5],
[85, 0, 0.1, 8],
[86, 0, 0.1, 1],
[87, 0, 0.1, 5],
[88, 0, 0.1, 6],
[89, 0, 0.1, 8],
[90, 0, 0.1, 8],
[91, 0, 0.1, 7],
[92, 0, 0.1, 7],
[93, 0, 0.1, 6],
[94, 0, 0.1, 6],
[95, 0, 0.1, 8],
[96, 0, 0.1, 4],
[97, 0, 0.1, 6],
[98, 0, 0.1, 6],
[99, 0, 0.1, 7],
[25, 1, 0.002, 5],
[26, 1, 0.002, 1],
[27, 1, 0.002, 1],
[28, 1, 0.002, 5],
[29, 1, 0.002, 6],
[30, 1, 0.002, 3],
[31, 1, 0.002, 9],
[32, 1, 0.002, 7],
[33, 1, 0.002, 1],
[34, 1, 0.002, 9],
[35, 1, 0.002, 1],
[36, 1, 0.002, 5],
[37, 1, 0.002, 3],
[38, 1, 0.002, 8],
[39, 1, 0.002, 6],
[40, 1, 0.002, 4],
[41, 1, 0.002, 2],
[42, 1, 0.002, 6],
[43, 1, 0.002, 4],
[44, 1, 0.002, 3],
[45, 1, 0.002, 5],
[46, 1, 0.002, 1],
[47, 1, 0.002, 3],
[48, 1, 0.002, 9],
[49, 1, 0.002, 8],
[50, 1, 0.1, 1],
[51, 1, 0.1, 1],
[52, 1, 0.1, 6],
[53, 1, 0.1, 7],
[54, 1, 0.1, 5],
[55, 1, 0.1, 5],
[56, 1, 0.1, 7],
[57, 1, 0.1, 1],
[58, 1, 0.1, 7],
[59, 1, 0.1, 2],
[60, 1, 0.1, 1],
[61, 1, 0.1, 2],
[62, 1, 0.1, 8],
[63, 1, 0.1, 6],
[64, 1, 0.1, 4],
[65, 1, 0.1, 8],
[66, 1, 0.1, 5],
[67, 1, 0.1, 4],
[68, 1, 0.1, 3],
[69, 1, 0.1, 1],
[70, 1, 0.1, 7],
[71, 1, 0.1, 8],
[72, 1, 0.1, 2],
[73, 1, 0.1, 1],
[74, 1, 0.1, 1],
[75, 1, 0.1, 9],
[76, 1, 0.1, 1],
[77, 1, 0.1, 8],
[78, 1, 0.1, 6],
[79, 1, 0.1, 1],
[80, 1, 0.1, 1],
[81, 1, 0.1, 7],
[82, 1, 0.1, 5],
[83, 1, 0.1, 8],
[84, 1, 0.1, 4],
[85, 1, 0.1, 7],
[86, 1, 0.1, 6],
[87, 1, 0.1, 4],
[88, 1, 0.1, 6],
[89, 1, 0.1, 9],
[90, 1, 0.1, 2],
[91, 1, 0.1, 4],
[92, 1, 0.1, 5],
[93, 1, 0.1, 1],
[94, 1, 0.1, 3],
[95, 1, 0.1, 4],
[96, 1, 0.1, 3],
[97, 1, 0.1, 1],
[98, 1, 0.1, 8],
[99, 1, 0.1, 5]]


IH_conn_list_inh=[[0, 1, -0.1, 4],
[1, 1, -0.1, 3],
[2, 1, -0.1, 4],
[3, 1, -0.1, 6],
[4, 1, -0.1, 9],
[5, 1, -0.1, 1],
[6, 1, -0.1, 7],
[7, 1, -0.1, 6],
[8, 1, -0.1, 5],
[9, 1, -0.1, 4],
[10, 1, -0.1, 5],
[11, 1, -0.1, 5],
[12, 1, -0.1, 9],
[13, 1, -0.1, 1],
[14, 1, -0.1, 6],
[15, 1, -0.1, 9],
[16, 1, -0.1, 4],
[17, 1, -0.1, 1],
[18, 1, -0.1, 8],
[19, 1, -0.1, 8],
[20, 1, -0.1, 1],
[21, 1, -0.1, 2],
[22, 1, -0.1, 7],
[23, 1, -0.1, 2],
[24, 1, -0.1, 2],
[25, 0, -0.1, 5],
[26, 0, -0.1, 6],
[27, 0, -0.1, 4],
[28, 0, -0.1, 1],
[29, 0, -0.1, 9],
[30, 0, -0.1, 9],
[31, 0, -0.1, 3],
[32, 0, -0.1, 3],
[33, 0, -0.1, 9],
[34, 0, -0.1, 7],
[35, 0, -0.1, 8],
[36, 0, -0.1, 2],
[37, 0, -0.1, 7],
[38, 0, -0.1, 4],
[39, 0, -0.1, 6],
[40, 0, -0.1, 8],
[41, 0, -0.1, 7],
[42, 0, -0.1, 2],
[43, 0, -0.1, 5],
[44, 0, -0.1, 3],
[45, 0, -0.1, 7],
[46, 0, -0.1, 7],
[47, 0, -0.1, 1],
[48, 0, -0.1, 8],
[49, 0, -0.1, 6]]

HH_conn_list_exc=[
    [0, 0, 0.0, 6],
    [1, 1, 0.0, 9]
    ]
HH_conn_list_inh=[
    [0, 1, 1.0, 4],
    [1, 0, 1.0, 4]
    ]

scalar = 1000
for i in IH_conn_list_exc:
    i[2] = i[2]*scalar

for i in IH_conn_list_inh:
    i[2] = i[2]*-scalar

for i in HH_conn_list_exc:
    i[2] = i[2]*scalar

for i in HH_conn_list_inh:
    i[2] = i[2]*scalar

# Create Input-to-Hidden projections
synapse_inp_to_hidden_exc = p.Projection(
    pop_input, pop_hidden, p.FromListConnector(IH_conn_list_exc),
    p.StaticSynapse(), receptor_type="excitatory")
synapse_inp_to_hidden_inh = p.Projection(
    pop_input, pop_hidden, p.FromListConnector(IH_conn_list_inh),
    p.StaticSynapse(), receptor_type="inhibitory")

# Create Hidden-to-Hidden (recurrent) projections
synapse_hidden_to_hidden_exc = p.Projection(
    pop_hidden, pop_hidden, p.FromListConnector(HH_conn_list_exc),
    p.StaticSynapse(), receptor_type="excitatory")

synapse_hidden_to_hidden_inh = p.Projection(
    pop_hidden, pop_hidden, p.FromListConnector(HH_conn_list_inh),
    p.StaticSynapse(), receptor_type="inhibitory")

# # Create Hidden-to-Output projections
# synapse_hidden_to_hidden = p.Projection(
#     pop_hidden, pop_output, p.FromListConnector(),
#     p.StaticSynapse(), receptor_type="excitatory")


#### Run Simulation ####
pop_input.record('all')
pop_hidden.record("all")
p.run(runtime)

pre_spikes = pop_input.get_data('spikes')
hidden_data = pop_hidden.get_data()

# Plot
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(pre_spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
    # plot data for postsynaptic neuron
    Panel(hidden_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[pop_hidden.label], yticks=True, xlim=(0, runtime)),
    Panel(hidden_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime)),
)


n=1
for n in [0,1]:

    for i in hidden_data.segments[0].spiketrains[0]:
        print i.magnitude

    print "\n\n\n\n\n"
    print "*************************************"

    for i in hidden_data.segments[0].filter(name='gsyn_inh')[0]:
        print i.magnitude[n]

    print "\n\n\n\n\n"
    print "*************************************"

    for i in hidden_data.segments[0].filter(name='v')[0]:
        print i.magnitude[n]

    print "\n\n\n\n\n"
    print "*************************************"

    for i in hidden_data.segments[0].filter(name='gsyn_exc')[0]:
        print i.magnitude[n]




plt.show()
p.end()


