

//TODO

/*

given the folder take every 200 scans
merge them into a point cloud
voxelize the point cloud with 10 cm leaf size
for each point find the stable normal - e.g. using 10 neighbours and LS fitting plane 
for vux points that have a stable normal - search the NN in the ref map and fit planes there
estimate point to plane cost function for each point 

keep track of 1000 or something points with normals

export a text file with these errors [scan_name, errors]

estimate mean, median, RMSE, stdev of these errors, etc. 



*/