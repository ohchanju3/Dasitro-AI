[out:json][timeout:90];
area["name:ko"="서울특별시"]["boundary"="administrative"]->.seoul;
rel(area.seoul)["boundary"="administrative"]["admin_level"="6"];
out ids tags geom;
