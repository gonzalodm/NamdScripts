ff = open("initconds","r")
file = ff.readlines()
ff.close()

# unit convert
BOHR_2_ANG = 0.529177

# Get Atom and Temp.
for line in file:
    if "Natom" in line:
        natom = int(line.split()[-1])
    if "Temp" in line:
        temp  = float(line.split()[-1])

print("Number of Atoms:",natom)
print("Temperature Ensemble [K]:",temp)

# Read geom and velocs
read = False
code = -1
for ii in range(len(file)):
    line = file[ii]
    if "Equilibrium" in line:
        read = True
        code = 0
        continue
    elif "Index" in line:
        read = True
        code = int(line.split()[-1])
        continue
    if "Atoms" in line:
        continue
    if read:
        # Open output files
        nameX = "geom_"  + str(code)
        nameXYZ = "geom_"  + str(code) + ".xyz"
        nameV = "veloc_" + str(code)
        fx = open(nameX,"w")
        fxyz = open(nameXYZ, "w")
        fv = open(nameV,"w")

        fxyz.write("%i\n" % natom)
        fxyz.write("\n")
        for jj in range(natom):
            idx = ii + jj
            line = file[idx].split()
            output = " ".join(line[:6]) + "\n"
            fx.write(output)
            output = " ".join(line[6:]) + "\n"
            fv.write(output)

            x = float(line[2]) * BOHR_2_ANG
            y = float(line[3]) * BOHR_2_ANG
            z = float(line[4]) * BOHR_2_ANG
            arr = [line[0],str(x),str(y),str(z)]
            output = " ".join(arr) + "\n"
            fxyz.write(output)
        read = False
        ii += natom






