ff = open("initconds","r")
file = ff.readlines()
ff.close()

# Get Atom and Temp.
for line in file:
    if "Natom" in line:
        natom = int(line.split()[-1])
    if "Temp" in line:
        temp  = float(line.split()[-1])

print("Number of Atoms:",natom)
print("Temperature Ensemble [K]:",temp)

ftraj = open("final_traj.xyz","w")

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
        nameV = "veloc_" + str(code)
        fx = open(nameX,"w")
        fv = open(nameV,"w")

        ftraj.write("%i\n" % natom)
        ftraj.write("\n")
        for jj in range(natom):
            idx = ii + jj
            line = file[idx].split()
            output = " ".join(line[:6]) + "\n"
            fx.write(output)
            output = " ".join(line[6:]) + "\n"
            fv.write(output)

            arr = [line[0],line[2],line[3],line[4]]
            output = " ".join(arr) + "\n"
            ftraj.write(output)
        read = False
        ii += natom






