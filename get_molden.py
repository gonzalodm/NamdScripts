import numpy as np
import argparse
import MDAnalysis as mda

# Convertion factor from hartree to cm^-1
HA_TO_CM1 = 219474.63
# Convertion factor from angstron to bohr
AN_TO_BO  = 1.88973

def getSingleTag(prop,file):
    arr = []
    for ii in range(len(file)):
        if prop in file[ii]:
            tmp = file[ii].split()[-1].split(':')[2:]
            # Fix for scalar value
            if tmp[0] == '0':
                tmp[0] = '1'
                tmp[1] = '1'

            # Get the dimensions of the arrays
            dimF = int(tmp[0])
            dimA = [int(ele) for ele in tmp[1].split(',')]
            dimA.reverse()

            # Checking
            if len(tmp) != 2:
                print('Error in Reading Tag %s' % (prop))
                exit(-1)
            if dimF != len(dimA):
                print('Error in Dimension of Tag %s' % (prop))
                exit(-1)

            # Read Array
            how_many_lines = np.prod(dimA) // 3 + (0 if np.prod(dimA) % 3 == 0 else 1)
            tmp = []
            for jj in range(ii+1,ii+1+how_many_lines):
                line = file[jj].split()
                for kk in range(len(line)):
                    tmp.append(float(line[kk]))

            arr = np.array(tmp).reshape(dimA)

            # Exit from loop after found prop
            break

    return arr

def readVibrations(vibfile,nAtoms):
    freqs = getSingleTag('frequencies',vibfile) * HA_TO_CM1
    modes = getSingleTag('eigenmodes',vibfile)
    # Check dimensions
    if freqs.shape[0] != 3*nAtoms or modes.shape[0] != 3*nAtoms or modes.shape[1] != 3*nAtoms:
        print('Dimensions in vibration file are wrong!..')
        exit(-1)
    # Set right shape of modes
    modes = modes.reshape(modes.shape[0],nAtoms,3)

    # clean negative freqs and set to zero the first 6.
    for ii in range(freqs.shape[0]):
        if freqs[ii] < 0. or ii < 6:
            freqs[ii] = 0.
    return freqs, modes



def main():
    # Init arguments
    parser = argparse.ArgumentParser(prog='get_molden.py', description='Make a molden file from the outputs of dftb+')

    # Geometry file
    parser.add_argument('-g','--geom', required = True, help='Geometry file in xyz format')

    # Vibrations file containing the normal modes and the frequencies
    parser.add_argument('-v','--vibr', required = True, help='Vibrations tag file')

    # Set arguments
    args = parser.parse_args()

    # Load geometry file
    try:
        univ = mda.Universe(args.geom)
    except IOError:
        raise IOError('File %s does not exist' % args.geom)

    # Load vibrations file
    try:
        ff = open(args.vibr,"r")
        vibrationsfile = ff.readlines()
        ff.close()
    except IOError:
        raise IOError('File %s does not exist' % args.vibr)

    # Set important variables and get the freq and modes
    atNames   = univ.atoms.names
    nAtoms    = univ.atoms.n_atoms
    positions = univ.atoms.positions * AN_TO_BO
    freq, modes = readVibrations(vibrationsfile,nAtoms)

    # Write final file
    try:
        fileout = open("freq.out.molden","w")
    except IOError:
        raise IOError('File freq.out.molden can not be created')

    fileout.write("[MOLDEN FORMAT]\n")
    fileout.write("[FREQ]\n")
    for ii in range(freq.shape[0]):
        fileout.write('%.2f\n' % freq[ii])

    fileout.write("[FR-COORD]\n")
    for ii in range(nAtoms):
        xyz = positions[ii]
        fileout.write(' %s  %.8f  %.8f  %.8f\n' % (atNames[ii],xyz[0],xyz[1],xyz[2]))

    fileout.write("[FR-NORM-COORD]\n")
    for ii in range(modes.shape[0]):
        fileout.write("vibration " + str(ii+1) + "\n")
        for jj in range(nAtoms):
            line = modes[ii,jj,:]
            if ii < 6:
                fileout.write("0.000000 0.000000 0.000000\n")
            else:
                xmod = "%f %f %f\n" % (line[0],line[1],line[2])
                fileout.write(xmod)


if __name__ == '__main__':
    main()
