#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2023 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
import shutil
# External Calls to MOLCAS
import subprocess as sp
# Command line arguments
import sys
# Regular expressions
import re
# debug print for dicts and arrays
import pprint
# sqrt and other math
import math
import cmath
# runtime measurement
import datetime
# copy of arrays of arrays
from copy import deepcopy
# parallel calculations
from multiprocessing import Pool
import time
# hostname
from socket import gethostname
# write debug traces when in pool threads
import traceback
# parse Python literals from input
import ast
import struct
# GDM: modules neede for dftb
# dftb input module
import hsd
import numpy as np

# ======================================================================= #

version = '3.0'
versiondate = datetime.date(2023, 4, 1)


changelogstring = '''
16.05.2018: INITIAL VERSION
- functionality as SHARC_GAUSSIAN.py, minus restricted triplets
- QM/MM capabilities in combination with TINKER
- AO overlaps computed by PyQuante (only up to f functions)

11.09.2018:
- added "basis_per_element", "basis_per_atom", and "hfexchange" keywords

03.10.2018:
Update for Orca 4.1:
- SOC for restricted singlets and triplets
- gradients for restricted triplets
- multigrad features
- orca_fragovl instead of PyQuante

16.10.2018:
Update for Orca 4.1, after revisions:
- does not work with Orca 4.0 or lower (orca_fragovl unavailable, engrad/pcgrad files)

11.10.2020:
- COBRAMM can be used for QM/MM calculations
'''

# ======================================================================= #
# holds the system time when the script was started
starttime = datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG = False
PRINT = True

# hash table for conversion of multiplicity to the keywords used in MOLCAS
IToMult = {
    1: 'Singlet',
    2: 'Doublet',
    3: 'Triplet',
    4: 'Quartet',
    5: 'Quintet',
    6: 'Sextet',
    7: 'Septet',
    8: 'Octet',
    9: '9-et',
    10: '10-et',
    11: '11-et',
    12: '12-et',
    13: '13-et',
    14: '14-et',
    15: '15-et',
    16: '16-et',
    17: '17-et',
    18: '18-et',
    19: '19-et',
    20: '20-et',
    21: '21-et',
    22: '22-et',
    23: '23-et',
    'Singlet': 1,
    'Doublet': 2,
    'Triplet': 3,
    'Quartet': 4,
    'Quintet': 5,
    'Sextet': 6,
    'Septet': 7,
    'Octet': 8
}

# hash table for conversion of polarisations to the keywords used in MOLCAS
IToPol = {
    0: 'X',
    1: 'Y',
    2: 'Z',
    'X': 0,
    'Y': 1,
    'Z': 2
}

# conversion factors
au2a = 0.529177211
rcm_to_Eh = 4.556335e-6
D2au = 0.393430307
au2eV = 27.2113987622
kcal_to_Eh = 0.0015936011

# =============================================================================================== #
# =============================================================================================== #
# =========================================== general routines ================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out

# ======================================================================= #


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
            sys.exit(13)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(14)

# ======================================================================= #


def isbinary(path):
    return (re.search(r':.* text', sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read()) is None)

# ======================================================================= #


def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))

# ======================================================================= #


def measuretime():
    '''Calculates the time difference between global variable starttime and the time of the call of measuretime.

    Prints the Runtime, if PRINT or DEBUG are enabled.

    Arguments:
    none

    Returns:
    1 float: runtime in seconds'''

    endtime = datetime.datetime.now()
    runtime = endtime - starttime
    hours = runtime.seconds // 3600
    minutes = runtime.seconds // 60 - hours * 60
    seconds = runtime.seconds % 60
    print('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days, hours, minutes, seconds))
    total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.e6
    return total_seconds

# ======================================================================= #


def removekey(d, key):
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d

# ======================================================================= #         OK


def containsstring(string, line):
    '''Takes a string (regular expression) and another string. Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

    a = re.search(string, line)
    if a:
        return True
    else:
        return False


# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def itmult(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return

# ======================================================================= #


def itnmstates(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return

# =============================================================================================== #
# =============================================================================================== #
# =========================================== print routines ==================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def printheader():
    '''Prints the formatted header of the log file. Prints version number and version date

    Takes nothing, returns nothing.'''

    print(starttime, gethostname(), os.getcwd())
    if not PRINT:
        return
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * 28 + 'SHARC - DFTB+ - Interface' + ' ' * 28 + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * 14 + 'Authors: Sebastian Mai, Lea Ibele, and Moritz Heindl' + ' ' * 14 + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * (36 - (len(version) + 1) // 2) + 'Version: %s' % (version) + ' ' * (35 - (len(version)) // 2) + '||\n'
    lens = len(versiondate.strftime("%d.%m.%y"))
    string += '||' + ' ' * (37 - lens // 2) + 'Date: %s' % (versiondate.strftime("%d.%m.%y")) + ' ' * (37 - (lens + 1) // 2) + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)
    if DEBUG:
        print(changelogstring)

# ======================================================================= #


def printQMin(QMin):

    if not PRINT:
        return
    print('==> QMin Job description for:\n%s' % (QMin['comment']))

    string = 'Mode:   '
    if 'init' in QMin:
        string += '\tINIT'
    if 'restart' in QMin:
        string += '\tRESTART'
    if 'samestep' in QMin:
        string += '\tSAMESTEP'
    if 'newstep' in QMin:
        string += '\tNEWSTEP'

    string += '\nTasks:  '
    if 'h' in QMin:
        string += '\tH'
    if 'soc' in QMin:
        string += '\tSOC'
    if 'dm' in QMin:
        string += '\tDM'
    if 'grad' in QMin:
        string += '\tGrad'
    if 'nacdr' in QMin:
        string += '\tNac(ddr)'
    if 'nacdt' in QMin:
        string += '\tNac(ddt)'
    if 'overlap' in QMin:
        string += '\tOverlap'
    if 'angular' in QMin:
        string += '\tAngular'
    if 'ion' in QMin:
        string += '\tDyson'
    if 'dmdr' in QMin:
        string += '\tDM-Grad'
    if 'socdr' in QMin:
        string += '\tSOC-Grad'
    if 'theodore' in QMin:
        string += '\tTheoDORE'
    if 'phases' in QMin:
        string += '\tPhases'
    print(string)

    string = 'States:        '
    for i in itmult(QMin['states']):
        string += '% 2i %7s  ' % (QMin['states'][i - 1], IToMult[i])
    print(string)

    string = 'Restricted:    '
    for i in itmult(QMin['states']):
        string += '%5s       ' % (QMin['jobs'][QMin['multmap'][i]]['restr'])
    print(string)

    string = 'Found Geo'
    if 'veloc' in QMin:
        string += ' and Veloc! '
    else:
        string += '! '
    string += 'NAtom is %i.\n' % (QMin['natom'])
    print(string)

    string = 'Geometry in Bohrs (%i atoms):\n' % QMin['natom']
    if DEBUG:
        for i in range(QMin['natom']):
            string += '%2s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][i][j + 1])
            string += '\n'
    else:
        for i in range(min(QMin['natom'], 5)):
            string += '%2s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][i][j + 1])
            string += '\n'
        if QMin['natom'] > 5:
            string += '..     ...     ...     ...\n'
            string += '%2s ' % (QMin['geo'][-1][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['geo'][-1][j + 1])
            string += '\n'
    print(string)

    if 'veloc' in QMin and DEBUG:
        string = ''
        for i in range(QMin['natom_orig']):
            string += '%s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['veloc'][i][j])
            string += '\n'
        print(string)

    if 'grad' in QMin:
        string = 'Gradients requested:   '
        for i in range(1, QMin['nmstates'] + 1):
            if i in QMin['grad']:
                string += 'X '
            else:
                string += '. '
        string += '\n'
        print(string)

    if 'nacdr' in QMin:
        string = 'Non-adiabatic couplings requested:\n'
        for i in range(1, QMin['nmstates'] + 1):
            for j in range(1, QMin['nmstates'] + 1):
                if [i, j] in QMin['nacdr'] or [j, i] in QMin['nacdr']:
                    string += 'X '
                else:
                    string += '. '
            string += '\n'
        print(string)

    print('State map:')
    pprint.pprint(QMin['statemap'])
    print

    for i in sorted(QMin):
        if not any([i == j for j in ['h', 'dm', 'soc', 'dmdr', 'socdr', 'theodore', 'geo', 'veloc', 'states', 'comment', 'grad', 'nacdr', 'ion', 'overlap', 'template', 'statemap', 'pointcharges', 'geo_orig', 'qmmm']]):
            if not any([i == j for j in ['ionlist']]) or DEBUG:
                string = i + ': '
                string += str(QMin[i])
                print(string)
    print('\n')
    sys.stdout.flush()


# ======================================================================= #
def printcomplexmatrix(matrix, states):
    '''Prints a formatted matrix. Zero elements are not printed, blocks of different mult and MS are delimited by dashes. Also prints a matrix with the imaginary parts, of any one element has non-zero imaginary part.

    Arguments:
    1 list of list of complex: the matrix
    2 list of integers: states specs'''

    nmstates = 0
    for i in range(len(states)):
        nmstates += states[i] * (i + 1)
    string = 'Real Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].real == 0.:
                string += ' ' * 11
            else:
                string += '% .3e ' % (matrix[istate][jstate].real)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    print(string)
    imag = False
    string = 'Imaginary Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].imag == 0.:
                string += ' ' * 11
            else:
                imag = True
                string += '% .3e ' % (matrix[istate][jstate].imag)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    string += '\n'
    if imag:
        print(string)

# ======================================================================= #


def printgrad(grad, natom, geo):
    '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

    Arguments:
    1 list of list of float: gradient
    2 integer: natom
    3 list of list: geometry specs'''

    string = ''
    iszero = True
    for atom in range(natom):
        if not DEBUG:
            if atom == 5:
                string += '...\t...\t     ...\t     ...\t     ...\n'
            if 5 <= atom < natom - 1:
                continue
        string += '%i\t%s\t' % (atom + 1, geo[atom][0])
        for xyz in range(3):
            if grad[atom][xyz] != 0:
                iszero = False
            g = grad[atom][xyz]
            if isinstance(g, float):
                string += '% .5f\t' % (g)
            elif isinstance(g, complex):
                string += '% .5f\t% .5f\t\t' % (g.real, g.imag)
        string += '\n'
    if iszero:
        print('\t\t...is identical zero...\n')
    else:
        print(string)

# ======================================================================= #


def printtheodore(matrix, QMin):
    string = '%6s ' % 'State'
    for i in QMin['template']['theodore_prop']:
        string += '%6s ' % i
    for i in range(len(QMin['template']['theodore_fragment'])):
        for j in range(len(QMin['template']['theodore_fragment'])):
            string += '  Om%1i%1i ' % (i + 1, j + 1)
    string += '\n' + '-------' * (1 + QMin['template']['theodore_n']) + '\n'
    istate = 0
    for imult, i, ms in itnmstates(QMin['states']):
        istate += 1
        string += '%6i ' % istate
        for i in matrix[istate - 1]:
            string += '%6.4f ' % i.real
        string += '\n'
    print(string)

# ======================================================================= #


def printQMout(QMin, QMout):
    '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout'''

    # if DEBUG:
    # pprint.pprint(QMout)
    if not PRINT:
        return
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    print('\n\n>>>>>>>>>>>>> Results\n')
    # Hamiltonian matrix, real or complex
    if 'h' in QMin or 'soc' in QMin:
        eshift = math.ceil(QMout['h'][0][0].real)
        print('=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift))
        matrix = deepcopy(QMout['h'])
        for i in range(nmstates):
            matrix[i][i] -= eshift
        printcomplexmatrix(matrix, states)
    # Dipole moment matrices
    if 'dm' in QMin:
        print('=> Dipole Moment Matrices:\n')
        for xyz in range(3):
            print('Polarisation %s:' % (IToPol[xyz]))
            matrix = QMout['dm'][xyz]
            printcomplexmatrix(matrix, states)
    # Gradients
    if 'grad' in QMin:
        print('=> Gradient Vectors:\n')
        istate = 0
        for imult, i, ms in itnmstates(states):
            print('%s\t%i\tMs= % .1f:' % (IToMult[imult], i, ms))
            printgrad(QMout['grad'][istate], natom, QMin['geo'])
            istate += 1

    # Nonadiabatic couplings
    if 'nacdr' in QMin:
        print('=> Analytical Non-adiabatic coupling vectors:\n')
        istate = 0
        for imult, i, msi in itnmstates(states):
            jstate = 0
            for jmult, j, msj in itnmstates(states):
                if imult == jmult and msi == msj:
                    print('%s\tStates %i - %i\tMs= % .1f:' % (IToMult[imult], i, j, msi))
                    printgrad(QMout['nacdr'][istate][jstate], natom, QMin['geo'])
                jstate += 1
            istate += 1

    # GDM: TODO: I deleted a lot of options that I think they are not needed.
    sys.stdout.flush()


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #         OK
def makecmatrix(a, b):
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    mat = [[complex(0., 0.) for i in range(a)] for j in range(b)]
    return mat

# ======================================================================= #         OK


def makermatrix(a, b):
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    mat = [[0. for i in range(a)] for j in range(b)]
    return mat


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def writeQMout(QMin, QMout, QMinfilename):
    '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out".

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout
    3 string: QMinfilename'''

    k = QMinfilename.find('.')
    if k == -1:
        outfilename = QMinfilename + '.out'
    else:
        outfilename = QMinfilename[:k] + '.out'
    if PRINT:
        print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
    string = ''
    if 'h' in QMin or 'soc' in QMin:
        string += writeQMoutsoc(QMin, QMout)
    if 'dm' in QMin:
        string += writeQMoutdm(QMin, QMout)
    if 'grad' in QMin:
        string += writeQMoutgrad(QMin, QMout)
    if 'nacdr' in QMin:
        string += writeQMoutnacana(QMin, QMout)
    if 'overlap' in QMin:
        string += writeQMoutnacsmat(QMin, QMout)
    if 'socdr' in QMin:
        string += writeQMoutsocdr(QMin, QMout)
    if 'dmdr' in QMin:
        string += writeQMoutdmdr(QMin, QMout)
    if 'ion' in QMin:
        string += writeQMoutprop(QMin, QMout)
    if 'phases' in QMin:
        string += writeQmoutPhases(QMin, QMout)
    string += writeQMouttime(QMin, QMout)
    outfile = os.path.join(QMin['pwd'], outfilename)
    writefile(outfile, string)
    return

# ======================================================================= #


def writeQMoutsoc(QMin, QMout):
    '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the SOC matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['h'][i][j].real, 9, 3), eformat(QMout['h'][i][j].imag, 9, 3))
        string += '\n'
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutdm(QMin, QMout):
    '''Generates a string with the Dipole moment matrices in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the DM matrices'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
    for xyz in range(3):
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (eformat(QMout['dm'][xyz][i][j].real, 9, 3), eformat(QMout['dm'][xyz][i][j].imag, 9, 3))
            string += '\n'
        # string+='\n'
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutgrad(QMin, QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the Gradient vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
        for atom in range(natom):
            for xyz in range(3):
                string += '%s ' % (eformat(QMout['grad'][i][atom][xyz], 9, 3))
            string += '\n'
        # string+='\n'
        i += 1
    string += '\n'
    return string

# =================================== #


def writeQMoutgradcobramm(QMin, QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the Gradient vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = len(QMout['pcgrad'][0])
    string = ''
    # string+='! %i Gradient Vectors (%ix%ix3, real)\n' % (3,nmstates,natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
        for atom in range(natom):
            for xyz in range(3):
                print((eformat(QMout['pcgrad'][i][atom][xyz], 9, 3)), i, atom)
                string += '%s ' % (eformat(QMout['pcgrad'][i][atom][xyz], 9, 3))
            string += '\n'
        # string+='\n'
        i += 1
    string += '\n'
    writefile("grad_charges", string)
# ======================================================================= #

def writeQMoutnacana(QMin, QMout):                                                                                                      
    '''Generates a string with the NAC vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the NAC vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += '%i %i ! %i %i %i %i %i %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s ' % (eformat(QMout['nacdr'][i][j][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            j += 1
        i += 1
    return string

def writeQMoutnacsmat(QMin, QMout):
    '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the transformation matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for j in range(nmstates):
        for i in range(nmstates):
            string += '%s %s ' % (eformat(QMout['overlap'][j][i].real, 9, 3), eformat(QMout['overlap'][j][i].imag, 9, 3))
        string += '\n'
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutdmdr(QMin, QMout):

    states = QMin['states']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Dipole moment derivatives (%ix%ix3x%ix3, real)\n' % (12, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            for ipol in range(3):
                string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i   pol %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms, ipol)
                for atom in range(natom):
                    for xyz in range(3):
                        string += '%s ' % (eformat(QMout['dmdr'][ipol][i][j][atom][xyz], 12, 3))
                    string += '\n'
                string += ''
            j += 1
        i += 1
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutsocdr(QMin, QMout):

    states = QMin['states']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Spin-Orbit coupling derivatives (%ix%ix3x%ix3, complex)\n' % (13, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += '%i %i ! m1 %i s1 %i ms1 %i   m2 %i s2 %i ms2 %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s %s ' % (eformat(QMout['socdr'][i][j][atom][xyz].real, 12, 3), eformat(QMout['socdr'][i][j][atom][xyz].imag, 12, 3))
            string += '\n'
            string += ''
            j += 1
        i += 1
    string += '\n'
    return string

# ======================================================================= #


def writeQMoutprop(QMin, QMout):

    nmstates = QMin['nmstates']

    # print property matrix (flag 11) for backwards compatibility
    string = ''
    string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
        string += '\n'
    string += '\n'

    # print property matrices (flag 20) in new format
    string += '! %i Property Matrices\n' % (20)
    string += '%i    ! number of property matrices\n' % (1)

    string += '! Property Matrix Labels (%i strings)\n' % (1)
    string += 'Dyson norms\n'

    string += '! Property Matrices (%ix%ix%i, complex)\n' % (1, nmstates, nmstates)
    string += '%i %i   ! Dyson norms\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
        string += '\n'
    string += '\n'

    return string

# ======================================================================= #


def writeQMoutTHEODORE(QMin, QMout):

    nmstates = QMin['nmstates']
    nprop = QMin['template']['theodore_n']
    if QMin['template']['qmmm']:
        nprop += len(QMin['qmmm']['MMEnergy_terms'])
    if nprop <= 0:
        return '\n'

    string = ''

    string += '! %i Property Vectors\n' % (21)
    string += '%i    ! number of property vectors\n' % (nprop)

    string += '! Property Vector Labels (%i strings)\n' % (nprop)
    descriptors = []
    if 'theodore' in QMin:
        for i in QMin['template']['theodore_prop']:
            descriptors.append('%s' % i)
            string += descriptors[-1] + '\n'
        for i in range(len(QMin['template']['theodore_fragment'])):
            for j in range(len(QMin['template']['theodore_fragment'])):
                descriptors.append('Om_{%i,%i}' % (i + 1, j + 1))
                string += descriptors[-1] + '\n'
    if QMin['template']['qmmm']:
        for label in sorted(QMin['qmmm']['MMEnergy_terms']):
            descriptors.append(label)
            string += label + '\n'

    string += '! Property Vectors (%ix%i, real)\n' % (nprop, nmstates)
    if 'theodore' in QMin:
        for i in range(QMin['template']['theodore_n']):
            string += '! TheoDORE descriptor %i (%s)\n' % (i + 1, descriptors[i])
            for j in range(nmstates):
                string += '%s\n' % (eformat(QMout['theodore'][j][i].real, 12, 3))
    if QMin['template']['qmmm']:
        for label in sorted(QMin['qmmm']['MMEnergy_terms']):
            string += '! QM/MM energy contribution (%s)\n' % (label)
            for j in range(nmstates):
                string += '%s\n' % (eformat(QMin['qmmm']['MMEnergy_terms'][label], 12, 3))
    string += '\n'

    return string

# ======================================================================= #


def writeQmoutPhases(QMin, QMout):

    string = '! 7 Phases\n%i ! for all nmstates\n' % (QMin['nmstates'])
    for i in range(QMin['nmstates']):
        string += '%s %s\n' % (eformat(QMout['phases'][i].real, 9, 3), eformat(QMout['phases'][i].imag, 9, 3))
    return string

# ======================================================================= #


def writeQMouttime(QMin, QMout):
    '''Generates a string with the quantum mechanics total runtime in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the runtime'''

    string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 9, 3))
    return string


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QM/MM ============================================= #
# =============================================================================================== #
# =============================================================================================== #

def prepare_QMMM(QMin, table_file):
    ''' creates dictionary with:
    MM coordinates (including connectivity and atom types)
    QM coordinates (including Link atom stuff)
    point charge data (including redistribution for Link atom neighbors)
    reorder arrays (for internal processing, all QM, then all LI, then all MM)

    is only allowed to read the following keys from QMin:
    geo
    natom
    QM/MM related infos from template
    '''

    table = readfile(table_file)


    # read table file
    print('===== Running QM/MM preparation ====')
    print('Reading table file ...         ', datetime.datetime.now())
    QMMM = {}
    QMMM['qmmmtype'] = []
    QMMM['atomtype'] = []
    QMMM['connect'] = []
    allowed = ['qm', 'mm']
    # read table file
    for iline, line in enumerate(table):
        s = line.split()
        if len(s) == 0:
            continue
        if not s[0].lower() in allowed:
            print('Not allowed QMMM-type "%s" on line %i!' % (s[0], iline + 1))
            sys.exit(15)
        QMMM['qmmmtype'].append(s[0].lower())
        QMMM['atomtype'].append(s[1])
        QMMM['connect'].append(set())
        for i in s[2:]:
            QMMM['connect'][-1].add(int(i) - 1)           # internally, atom numbering starts at 0
    QMMM['natom_table'] = len(QMMM['qmmmtype'])


    # list of QM and MM atoms
    QMMM['QM_atoms'] = []
    QMMM['MM_atoms'] = []
    for iatom in range(QMMM['natom_table']):
        if QMMM['qmmmtype'][iatom] == 'qm':
            QMMM['QM_atoms'].append(iatom)
        elif QMMM['qmmmtype'][iatom] == 'mm':
            QMMM['MM_atoms'].append(iatom)

    # make connections redundant and fill bond array
    print('Checking connection table ...  ', datetime.datetime.now())
    QMMM['bonds'] = set()
    for iatom in range(QMMM['natom_table']):
        for jatom in QMMM['connect'][iatom]:
            QMMM['bonds'].add(tuple(sorted([iatom, jatom])))
            QMMM['connect'][jatom].add(iatom)
    QMMM['bonds'] = sorted(list(QMMM['bonds']))


    # find link bonds
    print('Finding link bonds ...         ', datetime.datetime.now())
    QMMM['linkbonds'] = []
    QMMM['LI_atoms'] = []
    for i, j in QMMM['bonds']:
        if QMMM['qmmmtype'][i] != QMMM['qmmmtype'][j]:
            link = {}
            if QMMM['qmmmtype'][i] == 'qm':
                link['qm'] = i
                link['mm'] = j
            elif QMMM['qmmmtype'][i] == 'mm':
                link['qm'] = j
                link['mm'] = i
            link['scaling'] = {'qm': 0.3, 'mm': 0.7}
            link['element'] = 'H'
            link['atom'] = [link['element'], 0., 0., 0.]
            for xyz in range(3):
                link['atom'][xyz + 1] += link['scaling']['mm'] * QMin['geo'][link['mm']][xyz + 1]
                link['atom'][xyz + 1] += link['scaling']['qm'] * QMin['geo'][link['qm']][xyz + 1]
            QMMM['linkbonds'].append(link)
            QMMM['LI_atoms'].append(QMMM['natom_table'] - 1 + len(QMMM['linkbonds']))
            QMMM['atomtype'].append('999')
            QMMM['connect'].append(set([link['qm'], link['mm']]))


    # check link bonds
    mm_in_links = []
    qm_in_links = []
    mm_in_link_neighbors = []
    for link in QMMM['linkbonds']:
        mm_in_links.append(link['mm'])
        qm_in_links.append(link['qm'])
        for j in QMMM['connect'][link['mm']]:
            if QMMM['qmmmtype'][j] == 'mm':
                mm_in_link_neighbors.append(j)
    mm_in_link_neighbors.extend(mm_in_links)
    # no QM atom is allowed to be bonded to two MM atoms
    if not len(qm_in_links) == len(set(qm_in_links)):
        print('Some QM atom is involved in more than one link bond!')
        sys.exit(16)
    # no MM atom is allowed to be bonded to two QM atoms
    if not len(mm_in_links) == len(set(mm_in_links)):
        print('Some MM atom is involved in more than one link bond!')
        sys.exit(17)
    # no neighboring MM atoms are allowed to be involved in link bonds
    if not len(mm_in_link_neighbors) == len(set(mm_in_link_neighbors)):
        print('An MM-link atom is bonded to another MM-link atom!')
        sys.exit(18)


    # check geometry and connection table
    if not QMMM['natom_table'] == QMin['natom']:
        print('Number of atoms in table file does not match number of atoms in QMin!')
        sys.exit(19)


    # process MM geometry (and convert to angstrom!)
    QMMM['MM_coords'] = []
    for atom in QMin['geo']:
        QMMM['MM_coords'].append([atom[0]] + [i * au2a for i in atom[1:4]])
    for ilink, link in enumerate(QMMM['linkbonds']):
        QMMM['MM_coords'].append(['HLA'] + link['atom'][1:4])


    # create reordering dicts
    print('Creating reorder mappings ...  ', datetime.datetime.now())
    QMMM['reorder_input_MM'] = {}
    QMMM['reorder_MM_input'] = {}
    j = -1
    for i, t in enumerate(QMMM['qmmmtype']):
        if t == 'qm':
            j += 1
            QMMM['reorder_MM_input'][j] = i
    for ilink, link in enumerate(QMMM['linkbonds']):
        j += 1
        QMMM['reorder_MM_input'][j] = QMMM['natom_table'] + ilink
    for i, t in enumerate(QMMM['qmmmtype']):
        if t == 'mm':
            j += 1
            QMMM['reorder_MM_input'][j] = i
    for i in QMMM['reorder_MM_input']:
        QMMM['reorder_input_MM'][QMMM['reorder_MM_input'][i]] = i


    # process QM geometry (including link atoms), QM coords in bohr!
    QMMM['QM_coords'] = []
    QMMM['reorder_input_QM'] = {}
    QMMM['reorder_QM_input'] = {}
    j = -1
    for iatom in range(QMMM['natom_table']):
        if QMMM['qmmmtype'][iatom] == 'qm':
            QMMM['QM_coords'].append(deepcopy(QMin['geo'][iatom]))
            j += 1
            QMMM['reorder_input_QM'][iatom] = j
            QMMM['reorder_QM_input'][j] = iatom
    for ilink, link in enumerate(QMMM['linkbonds']):
        QMMM['QM_coords'].append(link['atom'])
        j += 1
        QMMM['reorder_input_QM'][-(ilink + 1)] = j
        QMMM['reorder_QM_input'][j] = -(ilink + 1)


    # process charge redistribution around link bonds
    # point charges are in input geometry ordering
    print('Charge redistribution ...      ', datetime.datetime.now())
    QMMM['charge_distr'] = []
    for iatom in range(QMMM['natom_table']):
        if QMMM['qmmmtype'][iatom] == 'qm':
            QMMM['charge_distr'].append([(0., 0)])
        elif QMMM['qmmmtype'][iatom] == 'mm':
            if iatom in mm_in_links:
                QMMM['charge_distr'].append([(0., 0)])
            else:
                QMMM['charge_distr'].append([(1., iatom)])
    for link in QMMM['linkbonds']:
        mm_neighbors = []
        for j in QMMM['connect'][link['mm']]:
            if QMMM['qmmmtype'][j] == 'mm':
                mm_neighbors.append(j)
        if len(mm_neighbors) > 0:
            factor = 1. / len(mm_neighbors)
            for j in QMMM['connect'][link['mm']]:
                if QMMM['qmmmtype'][j] == 'mm':
                    QMMM['charge_distr'][j].append((factor, link['mm']))

    # pprint.pprint(QMMM)
    return QMMM

# ======================================================================= #


def execute_tinker(QMin, ff_file_path):
    '''
    run tinker to get:
    * MM energy
    * MM gradient
    * point charges

    is only allowed to read the following keys from QMin:
    qmmm
    scratchdir
    savedir
    tinker
    '''

    QMMM = QMin['qmmm']

    # prepare Workdir
    WORKDIR = os.path.join(QMin['scratchdir'], 'TINKER')
    mkdir(WORKDIR)


    # key file
    # string='parameters %s\nQMMM %i\nQM %s\n' % (
    # ff_file_path,
    # QMMM['natom_table']+len(QMMM['linkbonds']),
    # ' '.join( [ str(QMMM['reorder_input_MM'][i]+1) for i in QMMM['QM_atoms'] ] )
    # )
    # if len(QMMM['linkbonds'])>0:
    # string+='LA %s\n' % (
    # ' '.join( [ str(QMMM['reorder_input_MM'][i]+1) for i in QMMM['LI_atoms'] ] ) )
    # string+='MM %s\n' % (
    # ' '.join( [ str(QMMM['reorder_input_MM'][i]+1) for i in QMMM['MM_atoms'] ] )  )
    # string+='\nDEBUG\n'
    # if len(QMMM['linkbonds'])>0:
    # string+='atom    999    99    HLA     "Hydrogen Link Atom"        1      1.008     0\n'
    # string+='\n'
    # filename=os.path.join(WORKDIR,'TINKER.key')
    # writefile(filename,string)


    print('Writing TINKER inputs ...      ', datetime.datetime.now())
    # key file
    string = 'parameters %s\nQMMM %i\n' % (ff_file_path, QMMM['natom_table'] + len(QMMM['linkbonds']))
    string += 'QM %i %i\n' % (-1, len(QMMM['QM_atoms']))
    if len(QMMM['linkbonds']) > 0:
        string += 'LA %s\n' % (
            ' '.join([str(QMMM['reorder_input_MM'][i] + 1) for i in QMMM['LI_atoms']]))
    string += 'MM %i %i\n' % (-(1 + len(QMMM['QM_atoms']) + len(QMMM['linkbonds'])),
                              QMMM['natom_table'] + len(QMMM['linkbonds']))
    # if DEBUG:
    # string+='\nDEBUG\n'
    if QMin['ncpu'] > 1:
        string += '\nOPENMP-THREADS %i\n' % QMin['ncpu']
    if len(QMMM['linkbonds']) > 0:
        string += 'atom    999    99    HLA     "Hydrogen Link Atom"        1      1.008     0\n'
    # string+='CUTOFF 1.0\n'
    string += '\n'
    filename = os.path.join(WORKDIR, 'TINKER.key')
    writefile(filename, string)


    # xyz/type/connection file
    string = '%i\n' % (len(QMMM['MM_coords']))
    for iatom_MM in range(len(QMMM['MM_coords'])):
        iatom_input = QMMM['reorder_MM_input'][iatom_MM]
        string += '% 5i  %3s  % 16.12f % 16.12f % 16.12f  %4s  %s\n' % (
            iatom_MM + 1,
            QMMM['MM_coords'][iatom_input][0],
            QMMM['MM_coords'][iatom_input][1],
            QMMM['MM_coords'][iatom_input][2],
            QMMM['MM_coords'][iatom_input][3],
            QMMM['atomtype'][iatom_input],
            ' '.join([str(QMMM['reorder_input_MM'][i] + 1) for i in sorted(QMMM['connect'][iatom_input])])
        )
    filename = os.path.join(WORKDIR, 'TINKER.xyz')
    writefile(filename, string)


    # communication file
    string = 'SHARC 0 -1\n'
    for iatom_MM in range(len(QMMM['MM_coords'])):
        iatom_input = QMMM['reorder_MM_input'][iatom_MM]
        string += '% 16.12f % 16.12f % 16.12f\n' % tuple(QMMM['MM_coords'][iatom_input][1:4])
    filename = os.path.join(WORKDIR, 'TINKER.qmmm')
    writefile(filename, string)


    # standard input file
    string = 'TINKER.xyz'
    filename = os.path.join(WORKDIR, 'TINKER.in')
    writefile(filename, string)


    # run TINKER
    runTINKER(WORKDIR, QMin['tinker'], QMin['savedir'], strip=False, ncpu=QMin['ncpu'])


    # read out TINKER
    filename = os.path.join(WORKDIR, 'TINKER.qmmm')
    output = readfile(filename)

    # check success
    if 'MMisOK' not in output[0]:
        print('TINKER run not successful!')
        sys.exit(20)

    # get MM energy (convert from kcal to Hartree)
    print('Searching MMEnergy ...         ', datetime.datetime.now())
    QMMM['MMEnergy'] = float(output[1].split()[-1]) * kcal_to_Eh

    # get MM gradient (convert from kcal/mole/A to Eh/bohr)
    print('Searching MMGradient ...       ', datetime.datetime.now())
    QMMM['MMGradient'] = {}
    for line in output:
        if 'MMGradient' in line:
            s = line.split()
            iatom_MM = int(s[1]) - 1
            iatom_input = QMMM['reorder_MM_input'][iatom_MM]
            grad = [float(i) * kcal_to_Eh * au2a for i in s[2:5]]
            QMMM['MMGradient'][iatom_input] = grad
        if 'MMq' in line:
            break

    # get MM point charges
    print('Searching MMpc_raw ...         ', datetime.datetime.now())
    QMMM['MMpc_raw'] = {}
    for i in range(QMMM['natom_table']):
        QMMM['MMpc_raw'][i] = 0.
    iline = 0
    while True:
        iline += 1
        line = output[iline]
        if 'MMq' in line:
            break
    iatom_MM = len(QMMM['QM_atoms']) + len(QMMM['LI_atoms']) - 1
    while True:
        iline += 1
        iatom_MM += 1
        line = output[iline]
        if 'NMM' in line:
            break
        s = line.split()
        q = float(s[-1])
        QMMM['MMpc_raw'][QMMM['reorder_MM_input'][iatom_MM]] = q

    # compute actual charges (including redistribution)
    print('Redistributing charges ...     ', datetime.datetime.now())
    QMMM['MMpc'] = {}
    for i in range(QMMM['natom_table']):
        s = 0.
        for factor, iatom in QMMM['charge_distr'][i]:
            s += factor * QMMM['MMpc_raw'][iatom]
        QMMM['MMpc'][i] = s

    # make list of pointcharges without QM atoms
    print('Finalizing charges ...         ', datetime.datetime.now())
    QMMM['pointcharges'] = []
    QMMM['reorder_pc_input'] = {}
    ipc = 0
    for iatom_input in QMMM['MM_atoms']:
        atom = QMMM['MM_coords'][iatom_input]
        q = QMMM['MMpc'][iatom_input]
        QMMM['pointcharges'].append(atom[1:4] + [q])
        QMMM['reorder_pc_input'][ipc] = iatom
        ipc += 1






    # Get energy components from standard out (debug print)
    filename = os.path.join(WORKDIR, 'TINKER.out')
    output = readfile(filename)
    QMMM['MMEnergy_terms'] = {}
    for line in output:
        if 'kcal/mol' in line:
            s = line.split()
            # QMMM['MMEnergy_terms'][s[0]]=float(s[-2])*kcal_to_Eh
            QMMM['MMEnergy_terms'][s[0]] = float(s[2])

    print('====================================')
    print('\n')

    # DONE! Final results:
    # QMMM['MMEnergy']
    # QMMM['MMGradient']
    # QMMM['MMpc']
    # QMMM['QM_coords']
    # QMMM['reorder_input_QM']
    # QMMM['reorder_QM_input']

    # print('='*60)
    # print('E:',QMMM['MMEnergy'])
    # print('Grad:')
    # pprint.pprint( QMMM['MMGradient'] )
    # print('MM coord:')
    # print str(QMMM['natom_table']) +'\n'
    # for atom in QMMM['MM_coords']:
    # print atom[0], atom[1], atom[2], atom[3]
    # print('QM coord:')
    # print str(len(QMMM['QM_coords'])) +'\n'
    # for atom in QMMM['QM_coords']:
    # print atom[0], atom[1]*au2a, atom[2]*au2a, atom[3]*au2a
    # print('MM pc:')
    # for iatom,atom in enumerate(QMMM['MM_coords']):
    # q=QMMM['MMpc'][iatom]
    # if q!=0.:
    # print atom[1],atom[2],atom[3],q
    # pprint.pprint(QMMM['MMEnergy_terms'])
    # print('='*60)

    # pprint.pprint(QMMM)
    # sys.exit(21)
    return QMMM

# ======================================================================= #


def coords_same(coord1, coord2):
    thres = 1e-5
    s = 0.
    for i in range(3):
        s += (coord1[i] - coord2[i])**2
    s = math.sqrt(s)
    return s <= thres

# ======================================================================= #


def runTINKER(WORKDIR, tinker, savedir, strip=False, ncpu=1):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = os.path.join(tinker, 'bin', 'tkr2qm_s') + ' '
    string += ' < TINKER.in'
    os.environ['OMP_NUM_THREADS'] = str(ncpu)
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    stdoutfile = open(os.path.join(WORKDIR, 'TINKER.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'TINKER.err'), 'w')
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(22)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    if DEBUG and runerror != 0:
        copydir = os.path.join(savedir, 'debug_TINKER_stdout')
        if not os.path.isdir(copydir):
            mkdir(copydir)
        outfile = os.path.join(WORKDIR, 'TINKER.out')
        tofile = os.path.join(copydir, "TINKER_problems.out")
        shutil.copy(outfile, tofile)
        print('Error in %s! Copied TINKER output to %s' % (WORKDIR, tofile))
    os.chdir(prevdir)
    if strip and not DEBUG and runerror == 0:
        stripWORKDIR(WORKDIR)
    return runerror

# ======================================================================= #


def transform_QM_QMMM(QMin, QMout):

    # Meta data
    QMin['natom'] = QMin['natom_orig']
    QMin['geo'] = QMin['geo_orig']

    # Hamiltonian
    if 'h' in QMout:
        for i in range(QMin['nmstates']):
            QMout['h'][i][i] += QMin['qmmm']['MMEnergy']

    # Gradients
    if 'grad' in QMout:
        nmstates = QMin['nmstates']
        natom = QMin['natom_orig']
        grad = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
        # QM gradient
        for iqm in QMin['qmmm']['reorder_QM_input']:
            iqmmm = QMin['qmmm']['reorder_QM_input'][iqm]
            if iqmmm < 0:
                ilink = -iqmmm - 1
                link = QMin['qmmm']['linkbonds'][ilink]
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][link['qm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['qm']
                        grad[istate][link['mm']][ixyz] += QMout['grad'][istate][iqm][ixyz] * link['scaling']['mm']
            else:
                for istate in range(nmstates):
                    for ixyz in range(3):
                        grad[istate][iqmmm][ixyz] += QMout['grad'][istate][iqm][ixyz]
        # PC gradient
        for iqm, iqmmm in enumerate(QMin['qmmm']['MM_atoms']):
            for istate in range(nmstates):
                for ixyz in range(3):
                    grad[istate][iqmmm][ixyz] += QMout['pcgrad'][istate][iqm][ixyz]
        # MM gradient
        for iqmmm in range(QMin['qmmm']['natom_table']):
            for istate in range(nmstates):
                for ixyz in range(3):
                    grad[istate][iqmmm][ixyz] += QMin['qmmm']['MMGradient'][iqmmm][ixyz]
        QMout['grad'] = grad

    # pprint.pprint(QMout)
    return QMin, QMout


# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO readQMin =========================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def checkscratch(SCRATCHDIR):
    '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR'''

    exist = os.path.exists(SCRATCHDIR)
    if exist:
        isfile = os.path.isfile(SCRATCHDIR)
        if isfile:
            print('$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR))
            sys.exit(23)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print('Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR))
            sys.exit(24)

# ======================================================================= #


def removequotes(string):
    if string.startswith("'") and string.endswith("'"):
        return string[1:-1]
    elif string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    else:
        return string

# ======================================================================= #


def getDftbVersion(path):
    # run dftb+ with nonexisting file
    string = os.path.join(path, 'dftb+')
    try:
        proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(25)
    comm = proc.communicate()[0].decode()
    # find version string
    for line in comm.split('\n'):
        if 'DFTB+' in line and ('release' in line or 'version' in line):
            s = re.findall("\d+\.\d+", line)
            s = s[0].split('.')
            s = tuple([int(i) for i in s])
            return s
    print('Could not find DFTB+ version!')
    sys.exit(26)

# ======================================================================= #


def getsh2Orcakey(sh2Orca, key):
    i = -1
    while True:
        i += 1
        try:
            line = re.sub('#.*$', '', sh2Orca[i])
        except IndexError:
            break
        line = line.split(None, 1)
        if line == []:
            continue
        if key.lower() in line[0].lower():
            return line
    return ['', '']

# ======================================================================= #


#GDM: TODO: Change the name to get_sh2Dftb_environ
def get_sh2Dftb_environ(sh2Orca, key, environ=True, crucial=True):
    line = getsh2Orcakey(sh2Orca, key)
    if line[0]:
        LINE = line[1]
        LINE = removequotes(LINE).strip()
    else:
        if environ:
            LINE = os.getenv(key.upper())
            if not LINE:
                if crucial:
                    print('Either set $%s or give path to %s in DFTB.resources!' % (key.upper(), key.upper()))
                    sys.exit(27)
                else:
                    return None
        else:
            if crucial:
                print('Give path to %s in DFTB.resources!' % (key.upper()))
                sys.exit(28)
            else:
                return None
    LINE = os.path.expandvars(LINE)
    LINE = os.path.expanduser(LINE)
    if containsstring(';', LINE):
        print("$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (key.upper(), key.upper()))
        sys.exit(29)
    return LINE

# ======================================================================= #


def get_pairs(QMinlines, i):
    nacpairs = []
    while True:
        i += 1
        try:
            line = QMinlines[i].lower()
        except IndexError:
            print('"keyword select" has to be completed with an "end" on another line!')
            sys.exit(30)
        if 'end' in line:
            break
        fields = line.split()
        try:
            nacpairs.append([int(fields[0]), int(fields[1])])
        except ValueError:
            print('"nacdr select" is followed by pairs of state indices, each pair on a new line!')
            sys.exit(31)
    return nacpairs, i

# ======================================================================= #         OK


def readQMin(QMinfilename):
    '''Reads the time-step dependent information from QMinfilename.

    Arguments:
    1 string: name of the QMin file

    Returns:
    1 dictionary: QMin'''


# --------------------------------------------- QM.in ----------------------------------

    QMinlines = readfile(QMinfilename)
    QMin = {}

    # Get natom
    try:
        natom = int(QMinlines[0])
    except ValueError:
        print('first line must contain the number of atoms!')
        sys.exit(32)
    QMin['natom'] = natom
    if len(QMinlines) < natom + 4:
        print('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task')
        sys.exit(33)

    # Save Comment line
    QMin['comment'] = QMinlines[1]

    # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
    QMin['geo'] = []
    QMin['veloc'] = []
    hasveloc = True
    for i in range(2, natom + 2):
        # only check line formatting for first 1000 atoms
        if i < 1000 and not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*', QMinlines[i]):
            print('Input file does not comply to xyz file format! Maybe natom is just wrong.')
            sys.exit(34)
        fields = QMinlines[i].split()
        fields[0] = fields[0].title()
        symb = fields[0]
        for j in range(1, 4):
            fields[j] = float(fields[j])
        QMin['geo'].append(fields[0:4])
        if len(fields) >= 7:
            for j in range(4, 7):
                fields[j] = float(fields[j])
            QMin['veloc'].append(fields[4:7])
        else:
            hasveloc = False
    if not hasveloc:
        QMin = removekey(QMin, 'veloc')


    # Parse remaining file
    i = natom + 1
    while i + 1 < len(QMinlines):
        i += 1
        line = QMinlines[i]
        line = re.sub('#.*$', '', line)
        if len(line.split()) == 0:
            continue
        key = line.lower().split()[0]
        if 'savedir' in key:
            args = line.split()[1:]
        else:
            args = line.lower().split()[1:]
        if key in QMin:
            print('Repeated keyword %s in line %i in input file! Check your input!' % (key, i + 1))
            continue  # only first instance of key in QM.in takes effect
        if len(args) >= 1 and args[0] == 'select':
            pairs, i = get_pairs(QMinlines, i)
            QMin[key] = pairs
        else:
            QMin[key] = args

    if 'unit' in QMin:
        if QMin['unit'][0] == 'angstrom':
            factor = 1. / au2a
        elif QMin['unit'][0] == 'bohr':
            factor = 1.
        else:
            print('Dont know input unit %s!' % (QMin['unit'][0]))
            sys.exit(35)
    else:
        factor = 1. / au2a

    for iatom in range(len(QMin['geo'])):
        for ixyz in range(3):
            QMin['geo'][iatom][ixyz + 1] *= factor

    # Calculate states, nstates, nmstates
    if 'states' not in QMin:
        print('Keyword "states" not given!')
        sys.exit(36)
    for i in range(len(QMin['states'])):
        QMin['states'][i] = int(QMin['states'][i])
    reduc = 0
    for i in reversed(QMin['states']):
        if i == 0:
            reduc += 1
        else:
            break
    for i in range(reduc):
        del QMin['states'][-1]
    nstates = 0
    nmstates = 0
    for i in range(len(QMin['states'])):
        nstates += QMin['states'][i]
        nmstates += QMin['states'][i] * (i + 1)
    QMin['nstates'] = nstates
    QMin['nmstates'] = nmstates

    # Various logical checks
    if 'states' not in QMin:
        print('Number of states not given in QM input file %s!' % (QMinfilename))
        sys.exit(37)

    # GDM: The interface SHARC-DFTB+ only works with singlets for the moment
    if len(QMin['states'])>1 and sum(QMin['states'][1:])>0:
        print('Currently, only singlet states are allowed!')
        sys.exit(38)

    # GDM: TODO: overlap is not needed in nacv, but for the moment we will include to run 
    #            a simple trajectory as a test
    #notpossibletasks = ['overlap','soc','dmdr','socdr','ion','theodore']
    notpossibletasks = ['soc','dmdr','socdr','ion','theodore']
    if any([i in QMin for i in notpossibletasks]):
        print('At least one of these Tasks are not implemented %s.' % notpossibletasks)
        sys.exit(39)

    if 'h' not in QMin:
        QMin['h'] = []

    if 'samestep' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(41)

    if 'restart' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(42)

    if 'phases' in QMin:
        QMin['overlap'] = []

    if 'overlap' in QMin and 'init' in QMin:
        print('"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"')
        sys.exit(43)

    if 'init' not in QMin and 'samestep' not in QMin and 'restart' not in QMin:
        QMin['newstep'] = []

    # Check for correct gradient list
    if 'grad' in QMin:
        if len(QMin['grad']) == 0 or QMin['grad'][0] == 'all':
            QMin['grad'] = [i + 1 for i in range(nmstates)]
        else:
            for i in range(len(QMin['grad'])):
                try:
                    QMin['grad'][i] = int(QMin['grad'][i])
                except ValueError:
                    print('Arguments to keyword "grad" must be "all" or a list of integers!')
                    sys.exit(47)
                if QMin['grad'][i] > nmstates:
                    print('State for requested gradient does not correspond to any state in QM input file state list!')
                    sys.exit(48)

    # GDM: Check for NACVs ######################################################
    #  TODO: also check that this should be incompatible with overlap
    if 'nacdr' in QMin:
        QMin['docicas'] = True # TODO: I don't know what is the reason for this variable
        if len(QMin['nacdr']) >= 1:
            nacpairs = QMin['nacdr']
            for i in range(len(nacpairs)):
                if nacpairs[i][0] > nmstates or nacpairs[i][1] > nmstates:
                    print('State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!')
                    sys.exit(48)
        else:
            QMin['nacdr'] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i)]
# --------------------------------------------- DFTB.resources ----------------------------------

    QMin['pwd'] = os.getcwd()

    # open DFTB.resources
    filename = 'DFTB.resources'
    if os.path.isfile(filename):
        sh2Dftb = readfile(filename)
    else:
        print('HINT: reading resources from SH2Dftb.inp')
        sh2Dftb = readfile('SH2Dftb.inp')

    # Set up scratchdir
    line = get_sh2Dftb_environ(sh2Dftb, 'scratchdir', False, False)
    if line is None:
        line = QMin['pwd'] + '/SCRATCHDIR/'
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    QMin['scratchdir'] = line
    link(QMin['scratchdir'], os.path.join(QMin['pwd'], 'SCRATCH'), False, False)

    # Set up savedir
    if 'savedir' in QMin:
        # savedir may be read from QM.in file
        line = QMin['savedir'][0]
    else:
        line = get_sh2Dftb_environ(sh2Dftb, 'savedir', False, False)
        if line is None:
            line = QMin['pwd'] + '/SAVEDIR/'
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    if 'init' in QMin:
        checkscratch(line)
    QMin['savedir'] = line
    link(QMin['savedir'], os.path.join(QMin['pwd'], 'SAVE'), False, False)

    # setup environment and Version for DFTB+
    QMin['dftbdir'] = get_sh2Dftb_environ(sh2Dftb, 'dftbdir')
    QMin['DftbVersion'] = getDftbVersion(QMin['dftbdir'])
    print('Detected DFTB+ version %s' % (str(QMin['DftbVersion'])))
    print('')
    if QMin['DftbVersion'] < (23, 1):
        print('This version of the SHARC-DFTB+ interface is only compatible to DFTB+ 23.1 or higher!')
        sys.exit(49)

    # debug option
    line = getsh2Orcakey(sh2Dftb, 'debug')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global DEBUG
            DEBUG = True

    # save_stuff option
    QMin['save_stuff'] = False
    line = getsh2Orcakey(sh2Dftb, 'save_stuff')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            QMin['save_stuff'] = True

    # print option
    line = getsh2Orcakey(sh2Dftb, 'no_print')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global PRINT
            PRINT = False

    # resources
    QMin['ncpu'] = 1
    line = getsh2Orcakey(sh2Dftb, 'ncpu')
    if line[0]:
        try:
            QMin['ncpu'] = int(line[1])
        except ValueError:
            print('Number of CPUs does not evaluate to numerical value!')
            sys.exit(50)
    if os.environ.get('NSLOTS') is not None:
        QMin['ncpu'] = int(os.environ.get('NSLOTS'))
        print('Detected $NSLOTS variable. Will use ncpu=%i' % (QMin['ncpu']))
    elif os.environ.get('SLURM_NTASKS_PER_NODE') is not None:
        QMin['ncpu'] = int(os.environ.get('SLURM_NTASKS_PER_NODE'))
        print('Detected $SLURM_NTASKS_PER_NODE variable. Will use ncpu=%i' % (QMin['ncpu']))
    QMin['ncpu'] = max(1, QMin['ncpu'])

    QMin['delay'] = 0.0
    line = getsh2Orcakey(sh2Dftb, 'delay')
    if line[0]:
        try:
            QMin['delay'] = float(line[1])
        except ValueError:
            print('Submit delay does not evaluate to numerical value!')
            sys.exit(51)

    QMin['schedule_scaling'] = 0.9
    line = getsh2Orcakey(sh2Dftb, 'schedule_scaling')
    if line[0]:
        try:
            x = float(line[1])
            if 0 < x <= 1.:
                QMin['schedule_scaling'] = x
        except ValueError:
            print('"schedule_scaling" does not evaluate to numerical value!')
            sys.exit(52)


    # initial MO guess settings
    # if neither keyword is present, the interface will reuse MOs from savedir, or let ADF generate a guess
    line = getsh2Orcakey(sh2Dftb, 'always_orb_init')
    if line[0]:
        QMin['always_orb_init'] = []
    line = getsh2Orcakey(sh2Dftb, 'always_guess')
    if line[0]:
        QMin['always_guess'] = []
    if 'always_orb_init' in QMin and 'always_guess' in QMin:
        print('Keywords "always_orb_init" and "always_guess" cannot be used together!')
        sys.exit(53)

    # memory
    QMin['memory'] = 100
    line = getsh2Orcakey(sh2Dftb, 'memory')
    if line[0]:
        QMin['memory'] = float(line[1])

    # truncation threshold
    # GDM: TODO: I think this is only used in overlap algorithm
    QMin['wfthres'] = 0.99
    line = getsh2Orcakey(sh2Dftb, 'wfthres')
    if line[0]:
        QMin['wfthres'] = float(line[1])

    # get the nooverlap keyword: no dets will be extracted if present
    line = getsh2Orcakey(sh2Dftb, 'nooverlap')
    if line[0]:
        QMin['nooverlap'] = []

    # neglected gradients
    QMin['neglected_gradient'] = 'zero'
    line = getsh2Orcakey(sh2Dftb, 'neglected_gradient')
    if line[0]:
        print("Neglected gradients are not allowed!.")
        sys.exit(57)

# --------------------------------------------- DFTB.template ----------------------------------
    QMin['template'] = loadTemplate('DFTB.template',QMin)

# --------------------------------------------- Logic ----------------------------------
    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(QMin['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    QMin['statemap'] = statemap

    # obtain the states to actually compute
    QMin['states_to_do'] = deepcopy(QMin['states'])

    # make the jobs
    jobs = {}
    if QMin['states_to_do'][0] > 0:
        jobs[1] = {'mults': [1], 'restr': True}
    QMin['jobs'] = jobs

    # make the multmap (mapping between multiplicity and job)
    # multmap[imult]=ijob
    # multmap[-ijob]=[imults]
    multmap = {}
    for ijob in jobs:
        job = jobs[ijob]
        for imult in job['mults']:
            multmap[imult] = ijob
        multmap[-(ijob)] = job['mults']
    multmap[1] = 1
    QMin['multmap'] = multmap

    # GDM: We will have only one job always since only Singlets are allow for the moment
    joblist = set()
    for i in jobs:
        joblist.add(i)
    joblist = sorted(joblist)
    QMin['joblist'] = joblist
    njobs = len(joblist)
    QMin['njobs'] = njobs

    # make the ground state map for every state
    gsmap = {}
    for i in range(QMin['nmstates']):
        m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
        gs = (m1, 1, ms1)
        job = QMin['multmap'][m1]
        if m1 == 3 and QMin['jobs'][job]['restr']:
            gs = (1, 1, 0.0)
        for j in range(QMin['nmstates']):
            m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
            if (m2, s2, ms2) == gs:
                break
        gsmap[i + 1] = j + 1
    QMin['gsmap'] = gsmap

    # get the set of states for which gradients actually need to be calculated
    gradmap = set()
    if 'grad' in QMin:
        for i in QMin['grad']:
            gradmap.add(tuple(statemap[i][0:2]))
    gradmap = list(gradmap)
    gradmap.sort()
    QMin['gradmap'] = gradmap

    # get the list of statepairs for NACdr calculation
    nacmap = set()
    if 'nacdr' in QMin:
        for i in QMin['nacdr']:                                                                                                         
            s1 = statemap[i[0]][0:2]
            s2 = statemap[i[1]][0:2]
            if s1[0] != s2[0] or s1 == s2:
                continue
            if s1[1] > s2[1]:
                continue
            nacmap.add(tuple(s1 + s2))
    nacmap = list(nacmap)
    nacmap.sort()
    QMin['nacmap'] = nacmap

# --------------------------------------------- File setup ---------------------------------- #

    # check for initial orbitals
    # GDM: TODO: This is necessary to restart dftb+, usefull when we want to calculate the
    # gradients for different states so it is not necessary to do the scf many times for the same conformation
    # Check along namd simulations
    initorbs = {}
    if 'always_guess' in QMin:
        QMin['initorbs'] = {}
    elif 'init' in QMin or 'always_orb_init' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['pwd'], 'charges.bin.init')
            if os.path.isfile(filename):
                initorbs[job] = filename
        for job in QMin['joblist']:
            filename = os.path.join(QMin['pwd'], 'charges.bin.%i.init' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
        if 'always_orb_init' in QMin and len(initorbs) < njobs:
            print('Initial orbitals missing for some jobs!')
            sys.exit(70)
        QMin['initorbs'] = initorbs
    elif 'newstep' in QMin:
        pass
       # GDM: TODO: dftb+ does not work with restart when we are using LC.
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'charges.bin.%i' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename + '.old'     # file will be moved to .old
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(71)
        QMin['initorbs'] = initorbs
    elif 'samestep' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'charges.bin.%i' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(72)
        QMin['initorbs'] = initorbs
    elif 'restart' in QMin:
        for job in QMin['joblist']:
            filename = os.path.join(QMin['savedir'], 'charges.bin.%i.old' % (job))
            if os.path.isfile(filename):
                initorbs[job] = filename
            else:
                print('File %s missing in savedir!' % (filename))
                sys.exit(73)
        QMin['initorbs'] = initorbs


    # make name for backup directory
    if 'backup' in QMin:
        backupdir = QMin['savedir'] + '/backup'
        backupdir1 = backupdir
        i = 0
        while os.path.isdir(backupdir1):
            i += 1
            if 'step' in QMin:
                backupdir1 = backupdir + '/step%s_%i' % (QMin['step'][0], i)
            else:
                backupdir1 = backupdir + '/calc_%i' % (i)
        QMin['backup'] = backupdir

    if DEBUG:
        print('======= DEBUG print for QMin =======')
        pprint.pprint(QMin)
        print('====================================')
    return QMin

# =============================================================================================== #
# =============================================================================================== #
# =========================================== Job Scheduling ==================================== #
# =============================================================================================== #
# =============================================================================================== #


def parallel_speedup(N, scaling):
    # computes the parallel speedup from Amdahls law
    # with scaling being the fraction of parallelizable work and (1-scaling) being the serial part
    return 1. / ((1 - scaling) + scaling / N)


def divide_slots(ncpu, ntasks, scaling):
    # this routine figures out the optimal distribution of the tasks over the CPU cores
    #   returns the number of rounds (how many jobs each CPU core will contribute to),
    #   the number of slots which should be set in the Pool,
    #   and the number of cores for each job.
    minpar = 1
    ntasks_per_round = ncpu // minpar
    if ncpu == 1:
        ntasks_per_round = 1
    ntasks_per_round = min(ntasks_per_round, ntasks)
    optimal = {}
    for i in range(1, 1 + ntasks_per_round):
        nrounds = int(math.ceil(float(ntasks) // i))
        ncores = ncpu // i
        optimal[i] = nrounds // parallel_speedup(ncores, scaling)
    # print optimal
    best = min(optimal, key=optimal.get)
    nrounds = int(math.ceil(float(ntasks) // best))
    ncores = ncpu // best

    cpu_per_run = [0 for i in range(ntasks)]
    if nrounds == 1:
        itask = 0
        for icpu in range(ncpu):
            cpu_per_run[itask] += 1
            itask += 1
            if itask >= ntasks:
                itask = 0
        nslots = ntasks
    else:
        for itask in range(ntasks):
            cpu_per_run[itask] = ncores
        nslots = ncpu // ncores
    # print nrounds,nslots,cpu_per_run
    return nrounds, nslots, cpu_per_run

# =============================================================================================== #


def generate_joblist(QMin):
    # pprint.pprint(QMin)

    # GDM: Warning. The program never should go inside this.
    if len(QMin['joblist']) != 1 or QMin['joblist'][0] != 1:
        print('Error in joblist..')
        sys.exit(100)

    # GDM: In this routine we will generate all the necessary jobs to run. All the calculations will run at the same time, since
    #      dftb+ is not able to do restart when we are using LC (which should be the default) is not necessary to run master before
    #      gradients. Essential all the jobs will calculate:
    #      master: excitation energies, ground state energy. This will carry out all the steps.
    #      grad_x: gradients, if x = 1 will be gradient on ground state or x > 2 will be the gradient on the x-1 excited state.
    #              This grad calculation could be different along the trajectory, since in every step we need to calculate a 
    #              couple of them.
    #      nacvs_1: calculate non-adiabatic coupling vectors between all the states.

    # Variables
    schedule = []
    QMin['nslots_pool'] = []

    # How many tasks will be running in parallel.
    ntasks = 1 # include master
    for grad in QMin['gradmap']:
        ntasks += 1

    # We add the nac calculationn
    if QMin['nacmap']:
        ntasks += 1
    nrounds, nslots, cpu_per_run = divide_slots(QMin['ncpu'],ntasks,QMin['schedule_scaling'])
    QMin['nslots_pool'].append(nslots)

    # Master Schedule
    QMin1 = deepcopy(QMin)
    QMin1['master'] = True
    QMin1['IJOB'] = 1
    remove = ['gradmap', 'ncpu']
    for r in remove:
        QMin1 = removekey(QMin1, r)
    QMin1['ncpu'] = cpu_per_run[0]
    schedule.append({})
    schedule[-1]['master_1'] = QMin1

    # Gradients Schedule
    icount = 1
    for grad in sorted(QMin['gradmap']):
        QMin1 = deepcopy(QMin)
        QMin1['IJOB'] = grad[1]
        remove = ['gradmap', 'ncpu', 'h', 'dm', 'always_guess', 'always_orb_init', 'init']
        for r in remove:
            QMin1 = removekey(QMin1, r)
        QMin1['gradmap'] = grad
        QMin1['ncpu'] = cpu_per_run[icount]
        schedule[-1]['grad_%i' % grad[1]] = QMin1
        icount += 1

    # Nacs Schedule
    if QMin['nacmap']:
        QMin1 = deepcopy(QMin)
        QMin1['IJOB'] = 1
        remove = ['gradmap', 'ncpu', 'h', 'dm', 'always_guess', 'always_orb_init', 'init']
        for r in remove:
            QMin1 = removekey(QMin1, r)
        QMin1['ncpu'] = cpu_per_run[icount]
        schedule[-1]['nacv_%i' % 1] = QMin1
        icount += 1
    return QMin, schedule


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Orca Job Execution ==================================== #
# =============================================================================================== #
# =============================================================================================== #

def runjobs(schedule, QMin):

    if 'newstep' in QMin:
        # GDM: TODO: this save the necessary files for restart, dftb+ does not work with this.
        moveOldFiles(QMin)

    # GDM: This launch all the jobs. schedule object will have (in principle) two elements. The first one should contain
    print('>>>>>>>>>>>>> Starting the DFTB+ job execution')
    errorcodes = {}
    for ijobset, jobset in enumerate(schedule):
        if not jobset:
            continue
        pool = Pool(processes=QMin['nslots_pool'][ijobset])
        for job in jobset:
            QMin1 = jobset[job]
            WORKDIR = os.path.join(QMin['scratchdir'], job)

            # GDM: the QMin1 has the input for every job to be done
            errorcodes[job] = pool.apply_async(run_calc, [WORKDIR, QMin1])
            time.sleep(QMin['delay'])
        pool.close()
        pool.join()

    for i in errorcodes:
        errorcodes[i] = errorcodes[i].get()
    j = 0
    string = 'Error Codes:\n'
    for i in errorcodes:
        string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
        j += 1
        if j == 4:
            j = 0
            string += '\n'
    print(string)
    if any((i != 0 for i in errorcodes.values())):
        print('Some subprocesses did not finish successfully!')
        print('See %s:%s for error messages in ORCA output.' % (gethostname(), QMin['scratchdir']))
        sys.exit(75)
    print

    if PRINT:
        print('>>>>>>>>>>>>> Saving files')
        starttime = datetime.datetime.now()
    # GDM: Here we saved all the important files to read in the master folder.
    for ijobset, jobset in enumerate(schedule):
        if not jobset:
            continue
        for ijob, job in enumerate(jobset):
            if 'master' in job:
                fromfile = os.path.join(QMin['scratchdir'], '%s/autotest.tag' % (job))
                if not os.path.isfile(fromfile):
                    print('File %s not found, cannot move to OLD!' % (fromfile))
                    sys.exit(77)
                tofile   = os.path.join(QMin['scratchdir'], '%s/autotest_master.tag' % (job))
                shutil.copy(fromfile, tofile)
            elif 'grad' in job:
                fromfile = os.path.join(QMin['scratchdir'], '%s/autotest.tag' % (job))
                if not os.path.isfile(fromfile):
                    print('File %s not found, cannot move to OLD!' % (fromfile))
                    sys.exit(77)
                tofile   = os.path.join(QMin['scratchdir'], 'master_1/autotest_%s.tag' % (job))
                shutil.copy(fromfile, tofile)
            elif 'nacv' in job:
                fromfile = os.path.join(QMin['scratchdir'], '%s/autotest.tag' % (job))
                if not os.path.isfile(fromfile):
                    print('File %s not found, cannot move to OLD!' % (fromfile))
                    sys.exit(77)
                tofile   = os.path.join(QMin['scratchdir'], 'master_1/autotest_nacvs.tag')
                shutil.copy(fromfile, tofile)

            # GDM: Original routine to save files for restart calculations
            if 'master' in job:
                WORKDIR = os.path.join(QMin['scratchdir'], job)
                if 'samestep' not in QMin:
                    saveFiles(WORKDIR, jobset[job])

    if PRINT:
        endtime = datetime.datetime.now()
        print('Saving Runtime: %s' % (endtime - starttime))
    print

    return errorcodes

# ======================================================================= #


def run_calc(WORKDIR, QMin):
    try:
        setupWORKDIR(WORKDIR, QMin)
        strip = True
        err = runDFTB(WORKDIR, QMin['dftbdir'], strip)
        # err=0
    except Exception as problem:
        print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
        traceback.print_exc()
        print('*' * 50 + '\n')
        raise problem

    return err

# ======================================================================= #

def loadTemplate(filetemplate,QMin):
    # Load template
    template = hsd.load(filetemplate)

    # Generate the format of the geometry
    geom = [[QMin["natom"]]]
    for iatom, atom in enumerate(QMin['geo']):
        lineatom = [atom[0]]
        lineatom.append(atom[1]*au2a)
        lineatom.append(atom[2]*au2a)
        lineatom.append(atom[3]*au2a)
        geom.append(lineatom)

    geom.append([])
    template["Geometry"] = {
            "xyzFormat": geom
    }
    # Add Variable to calculate Gradients and NACVs
    template["Analysis"] = {
            "CalculateForces":"Yes"
    }

    # Write Properties
    template["Options"] = {
            "WriteAutotestTag": "Yes"
    }

    return template

def modifyTemplate(QMin):
    # Master will perform a single point calculating the GS and ES energies.
    if 'master' in QMin:
        template = deepcopy(QMin['template'])
        template['Analysis']['CalculateForces'] = 'No'
        template['ExcitedState'] = {
                'Casida': {
                    'TammDancoff': 'Yes',
                    'NrOfExcitations': QMin['states'][0] - 1,
                    'Symmetry': 'Singlet',
                    'Diagonalizer': {
                        'Stratmann': {}
                    },
                    'WriteXplusY': 'Yes',
                    'WriteTransitionDipole': 'Yes',
                }
        }
    # Calculation of the gradients
    elif 'gradmap' in QMin:
        template = deepcopy(QMin['template'])
        if QMin['gradmap'] != (1, 1):
           template['ExcitedState'] = {
                   'Casida': {
                       'NrOfExcitations': QMin['states'][0] - 1,
                       'StateOfInterest': QMin['gradmap'][1] - 1,
                       'Symmetry': 'Singlet',
                       'Diagonalizer': {
                           'Stratmann': {}
                       },
                   }
           }
    # Calculation of the non-adiabatic couplings
    else:
        template = deepcopy(QMin['template'])
        template['ExcitedState'] = {
                'Casida': {
                    'NrOfExcitations': QMin['states'][0] - 1,
                    'StateOfInterest': 1,
                    'Symmetry': 'Singlet',
                    'Diagonalizer': {
                        'Stratmann': {}
                    },
                    'StateCouplings': '0\t'+str(QMin['states'][0]-1)
                }
        }
    return template


def setupWORKDIR(WORKDIR, QMin):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir
    # then put the dftb_in.hsd file

    # setup the directory
    mkdir(WORKDIR)

    # Create dftb input
    dftbinput = modifyTemplate(QMin)
    filename  = os.path.join(WORKDIR,'dftb_in.hsd')
    hsd.dump(dftbinput,filename)

    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(dftbinput)
        print('DFTB input written to: %s' % (filename))
        print('====================================================================')

    # GDM: here should be the code to copy the files necessary to use an initial guess.
    #      However in dftb+ restart calculation does not work when we are using LC, which probably will be
    #      the default option for NAMD.

    return

# ======================================================================= #


def writeORCAinput(QMin):
    # split gradmap into smaller chunks
    Nmax_gradlist = 255
    gradmaps = [sorted(QMin['gradmap'])[i:i + Nmax_gradlist] for i in range(0, len(QMin['gradmap']), Nmax_gradlist)]

    # make multi-job input
    string = ''
    for ichunk, chunk in enumerate(gradmaps):
        if ichunk >= 1:
            string += '\n\n$new_job\n\n%base "ORCA"\n\n'
        QMin_copy = deepcopy(QMin)
        QMin_copy['gradmap'] = chunk
        string += ORCAinput_string(QMin_copy)
    if not gradmaps:
        string += ORCAinput_string(QMin)
    return string


# ======================================================================= #
def ORCAinput_string(QMin):
    # pprint.pprint(QMin)

    # general setup
    job = QMin['IJOB']
    gsmult = QMin['multmap'][-job][0]
    restr = QMin['jobs'][job]['restr']
    charge = QMin['chargemap'][gsmult]

    # excited states to calculate
    states_to_do = QMin['states_to_do']
    for imult in range(len(states_to_do)):
        if not imult + 1 in QMin['multmap'][-job]:
            states_to_do[imult] = 0
    states_to_do[gsmult - 1] -= 1

    # do minimum number of states for gradient jobs
    if 'gradonly' in QMin:
        gradmult = QMin['gradmap'][0][0]
        gradstat = QMin['gradmap'][0][1]
        for imult in range(len(states_to_do)):
            if imult + 1 == gradmult:
                states_to_do[imult] = gradstat - (gradmult == gsmult)
            else:
                states_to_do[imult] = 0

    # number of states to calculate
    if restr:
        ncalc = max(states_to_do)
        # sing=states_to_do[0]>0
        trip = (len(states_to_do) >= 3 and states_to_do[2] > 0)
    else:
        ncalc = max(states_to_do)
        # mults_td=''

    # whether to do SOC
    # sopert=False
    # gscorr=False
    # if 'soc' in QMin:
        # if restr:
        # nsing=QMin['states'][0]
        # if len(QMin['states'])>=3:
        # ntrip=QMin['states'][2]
        # else:
        # ntrip=0
        # if nsing+ntrip>=2 and ntrip>=1:
        # sopert=True
        # if nsing>=1:
        # gscorr=True

    # gradients
    multigrad = False
    if 'grad' in QMin and QMin['gradmap']:
        dograd = True
        egrad = ()
        for grad in QMin['gradmap']:
            if not (gsmult, 1) == grad:
                egrad = grad
        # if len(QMin['gradmap'])>1:
        if QMin['OrcaVersion'] >= (4, 1):
            multigrad = True
            singgrad = []
            tripgrad = []
            for grad in QMin['gradmap']:
                if grad[0] == gsmult:
                    singgrad.append(grad[1] - 1)
                if grad[0] == 3 and restr:
                    tripgrad.append(grad[1])
    else:
        dograd = False

    # construct the input string
    string = ''

    # main line
    string += '! '

    keys = ['basis',
            'auxbasis',
            'functional',
            'dispersion',
            'ri',
            'keys']
    for i in keys:
        string += '%s ' % (QMin['template'][i])
    keys = ['nousesym']
    for i in keys:
        string += '%s ' % (i)

    if QMin['template']['grid']:
      string += 'grid%s ' % QMin['template']['grid']
    if QMin['template']['gridx']:
        string += 'gridx%s ' % QMin['template']['gridx']
# In this way, one can change grid on individual atoms:
# %method
# SpecialGridAtoms 26,15,-1,-4         # for element 26 and, for atom index 1 and 4 (cannot change on atom 0!)
# SpecialGridIntAcc 7,6,5,5            # size of grid
# end

    if dograd:
        string += 'engrad'

    string += '\n'

    # cpu cores
    if QMin['ncpu'] > 1 and 'AOoverlap' not in QMin:
        string += '%%pal\n  nprocs %i\nend\n\n' % (QMin['ncpu'])
    string += '%%maxcore %i\n\n' % (QMin['memory'])

    # basis sets
    if QMin['template']['basis_per_element']:
        string += '%basis\n'
        for i in QMin['template']['basis_per_element']:
            string += 'newgto %s "%s" end\n' % (i, QMin['template']['basis_per_element'][i])
        if not QMin['template']['ecp_per_element']:
            string += 'end\n\n'

    # ECP basis sets
    if QMin['template']['ecp_per_element']:
        if QMin['template']['basis_per_element']:
            for i in QMin['template']['ecp_per_element']:
                string += 'newECP %s "%s" end\n' % (i, QMin['template']['ecp_per_element'][i])
            string += 'end\n\n'
        else:
            print("ECP defined without additional basis. Not implemented.")

    # frozen core
    if QMin['frozcore'] > 0:
        string += '%%method\nfrozencore -%i\nend\n\n' % (2 * QMin['frozcore'])
    else:
        string += '%method\nfrozencore FC_NONE\nend\n\n'

    # hf exchange
    if QMin['template']['hfexchange'] >= 0.:
        # string+='%%method\nScalHFX = %f\nScalDFX = %f\nend\n\n' % (QMin['template']['hfexchange'],1.-QMin['template']['hfexchange'])
        string += '%%method\nScalHFX = %f\nend\n\n' % (QMin['template']['hfexchange'])

    # Range separation
    if QMin['template']['range_sep_settings']['do']:
        string += '''%%method
 RangeSepEXX True
 RangeSepMu %f
 RangeSepScal %f
 ACM %f, %f, %f\nend\n\n
''' % (QMin['template']['range_sep_settings']['mu'],
            QMin['template']['range_sep_settings']['scal'],
            QMin['template']['range_sep_settings']['ACM1'],
            QMin['template']['range_sep_settings']['ACM2'],
            QMin['template']['range_sep_settings']['ACM3']
       )

    # Intacc
    if QMin['template']['intacc'] > 0.:
        string += '''%%method
  intacc %3.1f\nend\n\n''' % (QMin['template']['intacc'])


    # Gaussian point charge scheme
    if 'cpcm' in QMin['template']['keys'].lower():
        string += '''%cpcm
  surfacetype vdw_gaussian\nend\n\n'''



    # excited states
    if ncalc > 0 and 'AOoverlap' not in QMin:
        string += '%tddft\n'
        if not QMin['template']['no_tda']:
            string += 'tda true\n'
        else:
            string += 'tda false\n'
        if QMin['template']['gridxc']:
            string += 'gridxc %s\n' % (QMin['template']['gridxc'])
        if 'theodore' in QMin:
            string += 'tprint 0.0001\n'
        if restr and trip:
            string += 'triplets true\n'
        string += 'nroots %i\n' % (ncalc)
        if restr and 'soc' in QMin:
            string += 'dosoc true\n'
            string += 'printlevel 3\n'
        # string+="dotrans all\n" #TODO
        if dograd:
            if multigrad:
                if singgrad:
                    string += 'sgradlist '
                    string += ','.join([str(i) for i in sorted(singgrad)])
                    string += '\n'
                if tripgrad:
                    string += 'tgradlist '
                    string += ','.join([str(i) for i in sorted(tripgrad)])
                    string += '\n'
            elif egrad:
                string += 'iroot %i\n' % (egrad[1] - (gsmult == egrad[0]))
        string += 'end\n\n'

    # output
    string += '%output\n'
    if 'AOoverlap' in QMin or 'ion' in QMin or 'theodore' in QMin:
        string += 'Print[ P_Overlap ] 1\n'
    if 'master' in QMin or 'theodore' in QMin:
        string += 'Print[ P_MOs ] 1\n'
    string += 'end\n\n'

    # scf
    string += '%scf\n'
    if 'AOoverlap' in QMin:
        string += 'maxiter 0\n'
    else:
        string += 'maxiter %i\n' % (QMin['template']['maxiter'])
    string += 'end\n\n'

    # rel
    if QMin['template']['picture_change']:
        string += '%rel\nPictureChange true\nend\n\n'


    # TODO: workaround
    # if 'soc' in QMin and 'grad' in QMin:
        # string+='%rel\nonecenter true\nend\n\n'

    # charge mult geom
    string += '%coords\nCtyp xyz\nunits bohrs\n'
    if 'AOoverlap' in QMin:
        string += 'charge %i\n' % (2. * charge)
    else:
        string += 'charge %i\n' % (charge)
    string += 'mult %i\n' % (gsmult)
    string += 'coords\n'
    for iatom, atom in enumerate(QMin['geo']):
        label = atom[0]
        string += '%4s %16.9f %16.9f %16.9f' % (label, atom[1], atom[2], atom[3])
        if iatom in QMin['template']['basis_per_atom']:
            string += ' newgto "%s" end' % (QMin['template']['basis_per_atom'][iatom])
        string += '\n'
    string += 'end\nend\n\n'

    # point charges
    if QMin['qmmm']:
        string += '%pointcharges "ORCA.pc"\n\n'
    elif QMin['template']['cobramm']:
        string += '%pointcharges "charge.dat"\n\n'
    if QMin['template']['paste_input_file']:
        string += '\n'
        for line in QMin['template']['paste_input_file']:
            string += line
        string += '\n'
    return string

# ======================================================================= #


def write_pccoord_file(QMin):
    string = '%i\n' % len(QMin['pointcharges'])
    for atom in QMin['pointcharges']:
        string += '%f %f %f %f\n' % (atom[3], atom[0], atom[1], atom[2])
    return string

# def write_pc_cobramm(QMin):
#    cobcharges=open('charges.dat', 'r')
#    charges=cobcharges.readlines()
#    string='%i\n' % len(charges)
#    for atom in charges:
#        string+='%f %f %f %f\n' % (atom[3],atom[0],atom[1],atom[2])
#    return string

# ======================================================================= #


def shorten_DIR(string):
    maxlen = 50
    front = 12
    if len(string) > maxlen:
        return string[0:front] + '...' + string[-(maxlen - 3 - front):]
    else:
        return string + ' ' * (maxlen - len(string))

# ======================================================================= #


def runDFTB(WORKDIR, dftbdir, strip=False):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = os.path.join(dftbdir, 'dftb+') + ' '
    string += 'dftb_in.hsd'
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    stdoutfile = open(os.path.join(WORKDIR, 'DFTB.log'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'DFTB.err'), 'w')
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(76)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    if strip and not DEBUG and runerror == 0:
        stripWORKDIR(WORKDIR)
    return runerror

# ======================================================================= #


def stripWORKDIR(WORKDIR):
    ls = os.listdir(WORKDIR)
    keep = ['dftb_in.hsd', 'DFTB.err$', 'DFTB.log$', 'charges.bin', 'EXC.DAT', 'NACV.DAT', 'autotest.tag', 'detailed.out','XplusY.DAT','TDP.DAT']
    for ifile in ls:
        delete = True
        for k in keep:
            if containsstring(k, ifile):
                delete = False
        if delete:
            rmfile = os.path.join(WORKDIR, ifile)
            if not DEBUG:
                os.remove(rmfile)

# ======================================================================= #


def moveOldFiles(QMin):
    # moves all relevant files in the savedir to old files (per job)
    # GDM: TODO: this is not necessary in fact since dftb+ does not make restart in LC
    if PRINT:
        print('>>>>>>>>>>>>> Moving old files')
    basenames = ['charges.bin','XplusY.DAT']
    if 'nooverlap' not in QMin:
        pass
        #basenames.append('mos')
    for job in QMin['joblist']:
        for base in basenames:
            fromfile = os.path.join(QMin['savedir'], '%s.%i' % (base, job))
            if not os.path.isfile(fromfile):
                print('File %s not found, cannot move to OLD!' % (fromfile))
                sys.exit(77)
            tofile = os.path.join(QMin['savedir'], '%s.%i.old' % (base, job))
            if PRINT:
                print(shorten_DIR(fromfile) + '   =>   ' + shorten_DIR(tofile))
            shutil.copy(fromfile, tofile)
    # moves all relevant files in the savedir to old files (per mult)
    basenames = []
# GDM: TODO: this is not needed
#   if 'nooverlap' not in QMin:
#       basenames = ['dets']
#   for job in itmult(QMin['states']):
#       for base in basenames:
#           fromfile = os.path.join(QMin['savedir'], '%s.%i' % (base, job))
#           if not os.path.isfile(fromfile):
#               print('File %s not found, cannot move to OLD!' % (fromfile))
#               sys.exit(78)
#           tofile = os.path.join(QMin['savedir'], '%s.%i.old' % (base, job))
#           if PRINT:
#               print(shorten_DIR(fromfile) + '   =>   ' + shorten_DIR(tofile))
#           shutil.copy(fromfile, tofile)

# GDM: TODO: this is also not needed
#   # also remove aoovl files if present
#   delete = ['AO_overl', 'AO_overl.mixed']
#   for f in delete:
#       rmfile = os.path.join(QMin['savedir'], f)
#       if os.path.isfile(rmfile):
#           os.remove(rmfile)
#           if PRINT:
#               print('rm ' + rmfile)
    print

# ======================================================================= #
# def saveGeometry(QMin):
    # string=''
    # for iatom,atom in enumerate(QMin['geo']):
    # label=atom[0]
    # string+='%4s %16.9f %16.9f %16.9f\n' % (label,atom[1],atom[2],atom[3])
    # filename=os.path.join(QMin['savedir'],'geom.dat')
    # writefile(filename,string)
    # if PRINT:
    # print shorten_DIR(filename)
    # return

# ======================================================================= #


def saveFiles(WORKDIR, QMin):
    # copy the bin files from master directories
    job = QMin['IJOB']
    basenames = ['charges.bin','XplusY.DAT']
    for file in basenames:
        fromfile = os.path.join(WORKDIR, file)
        if not os.path.isfile(fromfile):
           print('File %s not found, cannot move to OLD!' % (fromfile))
           sys.exit(77)
        tofile = os.path.join(QMin['savedir'], '%s.%i' % (file,job))
        shutil.copy(fromfile, tofile)
        if PRINT:
            print(shorten_DIR(tofile))

# ======================================================================= #


# ======================================================================= #
def get_MO_from_gbw(filename, QMin):

    # run orca_fragovl
    string = 'orca_fragovl %s %s' % (filename, filename)
    try:
        proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(80)
    comm = proc.communicate()[0].decode()
    data = comm.split('\n')
    # get size of matrix
    for line in reversed(data):
        # print line
        s = line.split()
        if len(s) >= 1:
            NAO = int(line.split()[0]) + 1
            break

    job = QMin['IJOB']
    restr = QMin['jobs'][job]['restr']

    # find MO block
    iline = -1
    while True:
        iline += 1
        if len(data) <= iline:
            print('MOs not found!')
            sys.exit(81)
        line = data[iline]
        if 'FRAGMENT A MOs MATRIX' in line:
            break
    iline += 3

    # formatting
    nblock = 6
    npre = 11
    ndigits = 16
    # default_pos=[14,30,46,62,78,94]
    default_pos = [npre + 3 + ndigits * i for i in range(nblock)]  # does not include shift

    # get coefficients for alpha
    NMO_A = NAO
    MO_A = [[0. for i in range(NAO)] for j in range(NMO_A)]
    for imo in range(NMO_A):
        jblock = imo // nblock
        jcol = imo % nblock
        for iao in range(NAO):
            shift = max(0, len(str(iao)) - 3)
            jline = iline + jblock * (NAO + 1) + iao
            line = data[jline]
            # fix too long floats in strings
            dots = [idx for idx, item in enumerate(line.lower()) if '.' in item]
            diff = [dots[i] - default_pos[i] - shift for i in range(len(dots))]
            if jcol == 0:
                pre = 0
            else:
                pre = diff[jcol - 1]
            post = diff[jcol]
            # fixed
            val = float(line[npre + shift + jcol * ndigits + pre: npre + shift + ndigits + jcol * ndigits + post])
            MO_A[imo][iao] = val
    iline += ((NAO - 1) // nblock + 1) * (NAO + 1)

    # coefficients for beta
    if not restr:
        NMO_B = NAO
        MO_B = [[0. for i in range(NAO)] for j in range(NMO_B)]
        for imo in range(NMO_B):
            jblock = imo // nblock
            jcol = imo % nblock
            for iao in range(NAO):
                shift = max(0, len(str(iao)) - 3)
                jline = iline + jblock * (NAO + 1) + iao
                line = data[jline]
                # fix too long floats in strings
                dots = [idx for idx, item in enumerate(line.lower()) if '.' in item]
                diff = [dots[i] - default_pos[i] - shift for i in range(len(dots))]
                if jcol == 0:
                    pre = 0
                else:
                    pre = diff[jcol - 1]
                post = diff[jcol]
                # fixed
                val = float(line[npre + shift + jcol * ndigits + pre: npre + shift + ndigits + jcol * ndigits + post])
                MO_B[imo][iao] = val


    NMO = NMO_A - QMin['frozcore']
    if restr:
        NMO = NMO_A - QMin['frozcore']
    else:
        NMO = NMO_A + NMO_B - 2 * QMin['frozcore']

    # make string
    string = '''2mocoef
header
 1
MO-coefficients from Orca
 1
 %i   %i
 a
mocoef
(*)
''' % (NAO, NMO)
    x = 0
    for imo, mo in enumerate(MO_A):
        if imo < QMin['frozcore']:
            continue
        for c in mo:
            if x >= 3:
                string += '\n'
                x = 0
            string += '% 6.12e ' % c
            x += 1
        if x > 0:
            string += '\n'
            x = 0
    if not restr:
        x = 0
        for imo, mo in enumerate(MO_B):
            if imo < QMin['frozcore']:
                continue
            for c in mo:
                if x >= 3:
                    string += '\n'
                    x = 0
                string += '% 6.12e ' % c
                x += 1
            if x > 0:
                string += '\n'
                x = 0
    string += 'orbocc\n(*)\n'
    x = 0
    for i in range(NMO):
        if x >= 3:
            string += '\n'
            x = 0
        string += '% 6.12e ' % (0.0)
        x += 1

    return string

# ======================================================================= #


def get_dets_from_cis(filename, QMin):

    # get general infos
    job = QMin['IJOB']
    restr = QMin['jobs'][job]['restr']
    mults = QMin['jobs'][job]['mults']
    gsmult = QMin['multmap'][-job][0]
    nstates_to_extract = deepcopy(QMin['states'])
    nstates_to_skip = [QMin['states_to_do'][i] - QMin['states'][i] for i in range(len(QMin['states']))]
    for i in range(len(nstates_to_extract)):
        if not i + 1 in mults:
            nstates_to_extract[i] = 0
            nstates_to_skip[i] = 0
        elif i + 1 == gsmult:
            nstates_to_extract[i] -= 1
    # print job,restr,mults,gsmult,nstates_to_extract

    # get infos from logfile
    logfile = os.path.join(os.path.dirname(filename), 'ORCA.log')
    data = readfile(logfile)
    infos = {}
    for iline, line in enumerate(data):
        if '# of contracted basis functions' in line:
            infos['nbsuse'] = int(line.split()[-1])
        if 'Orbital ranges used for CIS calculation:' in line:
            s = data[iline + 1].replace('.', ' ').split()
            infos['NFC'] = int(s[3])
            infos['NOA'] = int(s[4]) - int(s[3]) + 1
            infos['NVA'] = int(s[7]) - int(s[6]) + 1
            if restr:
                infos['NOB'] = infos['NOA']
                infos['NVB'] = infos['NVA']
            else:
                s = data[iline + 2].replace('.', ' ').split()
                infos['NOB'] = int(s[4]) - int(s[3]) + 1
                infos['NVB'] = int(s[7]) - int(s[6]) + 1

    if 'NOA' not in infos:
        nstates_onfile = 0
        charge = QMin['chargemap'][gsmult]
        nelec = float(QMin['Atomcharge'] - charge)
        infos['NOA'] = int(nelec / 2. + float(gsmult - 1) / 2.)
        infos['NOB'] = int(nelec / 2. - float(gsmult - 1) / 2.)
        infos['NVA'] = infos['nbsuse'] - infos['NOA']
        infos['NVB'] = infos['nbsuse'] - infos['NOB']
        infos['NFC'] = 0
    else:
        # get all info from cis file
        CCfile = open(filename, 'rb')
        nvec = struct.unpack('i', CCfile.read(4))[0]
        header = [struct.unpack('i', CCfile.read(4))[0] for i in range(8)]
        # print infos
        # print header
        if infos['NOA'] != header[1] - header[0] + 1:
            print('Number of orbitals in %s not consistent' % filename)
            sys.exit(82)
        if infos['NVA'] != header[3] - header[2] + 1:
            print('Number of orbitals in %s not consistent' % filename)
            sys.exit(83)
        if not restr:
            if infos['NOB'] != header[5] - header[4] + 1:
                print('Number of orbitals in %s not consistent' % filename)
                sys.exit(84)
            if infos['NVB'] != header[7] - header[6] + 1:
                print('Number of orbitals in %s not consistent' % filename)
                sys.exit(85)
        if QMin['template']['no_tda']:
            nstates_onfile = nvec // 2
        else:
            nstates_onfile = nvec


    # get ground state configuration
    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        occ_A = [3 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
    if not restr:
        occ_A = [1 for i in range(infos['NFC'] + infos['NOA'])] + [0 for i in range(infos['NVA'])]
        occ_B = [2 for i in range(infos['NFC'] + infos['NOB'])] + [0 for i in range(infos['NVB'])]
    occ_A = tuple(occ_A)
    if not restr:
        occ_B = tuple(occ_B)

    # get infos
    nocc_A = infos['NOA']
    nvir_A = infos['NVA']
    nocc_B = infos['NOB']
    nvir_B = infos['NVB']

    # get eigenvectors
    eigenvectors = {}
    for imult, mult in enumerate(mults):
        eigenvectors[mult] = []
        if mult == gsmult:
            # add ground state
            if restr:
                key = tuple(occ_A[QMin['frozcore']:])
            else:
                key = tuple(occ_A[QMin['frozcore']:] + occ_B[QMin['frozcore']:])
            eigenvectors[mult].append({key: 1.0})
        for istate in range(nstates_to_extract[mult - 1]):
            CCfile.read(40)
            dets = {}
            for iocc in range(header[0], header[1] + 1):
                for ivirt in range(header[2], header[3] + 1):
                    dets[(iocc, ivirt, 1)] = struct.unpack('d', CCfile.read(8))[0]
            if not restr:
                for iocc in range(header[4], header[5] + 1):
                    for ivirt in range(header[6], header[7] + 1):
                        dets[(iocc, ivirt, 2)] = struct.unpack('d', CCfile.read(8))[0]
            if QMin['template']['no_tda']:
                CCfile.read(40)
                for iocc in range(header[0], header[1] + 1):
                    for ivirt in range(header[2], header[3] + 1):
                        dets[(iocc, ivirt, 1)] += struct.unpack('d', CCfile.read(8))[0]
                        dets[(iocc, ivirt, 1)] /= 2.
                if not restr:
                    for iocc in range(header[4], header[5] + 1):
                        for ivirt in range(header[6], header[7] + 1):
                            dets[(iocc, ivirt, 2)] += struct.unpack('d', CCfile.read(8))[0]
                            dets[(iocc, ivirt, 2)] /= 2.

            # pprint.pprint(dets)
            # truncate vectors
            norm = 0.
            for k in sorted(dets, key=lambda x: dets[x]**2, reverse=True):
                factor = 1.
                if norm > factor * QMin['wfthres']:
                    del dets[k]
                    continue
                norm += dets[k]**2
            # pprint.pprint(dets)
            # create strings and expand singlets
            dets2 = {}
            if restr:
                for iocc, ivirt, dummy in dets:
                    # singlet
                    if mult == 1:
                        # alpha excitation
                        key = list(occ_A)
                        key[iocc] = 2
                        key[ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(0.5)
                        # beta excitation
                        key[iocc] = 1
                        key[ivirt] = 2
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)] * math.sqrt(0.5)
                    # triplet
                    elif mult == 3:
                        key = list(occ_A)
                        key[iocc] = 1
                        key[ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
            else:
                for iocc, ivirt, dummy in dets:
                    if dummy == 1:
                        key = list(occ_A + occ_B)
                        key[iocc] = 0
                        key[ivirt] = 1
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
                    elif dummy == 2:
                        key = list(occ_A + occ_B)
                        key[infos['NFC'] + nocc_A + nvir_A + iocc] = 0
                        key[infos['NFC'] + nocc_A + nvir_A + ivirt] = 2
                        dets2[tuple(key)] = dets[(iocc, ivirt, dummy)]
            # pprint.pprint(dets2)
            # remove frozen core
            dets3 = {}
            for key in dets2:
                problem = False
                if restr:
                    if any([key[i] != 3 for i in range(QMin['frozcore'])]):
                        problem = True
                else:
                    if any([key[i] != 1 for i in range(QMin['frozcore'])]):
                        problem = True
                    if any([key[i] != 2 for i in range(nocc_A + nvir_A + QMin['frozcore'], nocc_A + nvir_A + 2 * QMin['frozcore'])]):
                        problem = True
                if problem:
                    print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                    continue
                    # sys.exit(86)
                if restr:
                    key2 = key[QMin['frozcore']:]
                else:
                    key2 = key[QMin['frozcore']:QMin['frozcore'] + nocc_A + nvir_A] + key[nocc_A + nvir_A + 2 * QMin['frozcore']:]
                dets3[key2] = dets2[key]
            # pprint.pprint(dets3)
            # append
            eigenvectors[mult].append(dets3)
        # skip extra roots
        for istate in range(nstates_to_skip[mult - 1]):
            CCfile.read(40)
            for iocc in range(header[0], header[1] + 1):
                for ivirt in range(header[2], header[3] + 1):
                    CCfile.read(8)
            if not restr:
                for iocc in range(header[4], header[5] + 1):
                    for ivirt in range(header[6], header[7] + 1):
                        CCfile.read(8)
            if QMin['template']['no_tda']:
                CCfile.read(40)
                for iocc in range(header[0], header[1] + 1):
                    for ivirt in range(header[2], header[3] + 1):
                        CCfile.read(8)
                if not restr:
                    for iocc in range(header[4], header[5] + 1):
                        for ivirt in range(header[6], header[7] + 1):
                            CCfile.read(8)


    strings = {}
    for imult, mult in enumerate(mults):
        filename = os.path.join(QMin['savedir'], 'dets.%i' % mult)
        strings[filename] = format_ci_vectors(eigenvectors[mult])

    return strings

# ======================================================================= #


def format_ci_vectors(ci_vectors):

    # get nstates, norb and ndets
    alldets = set()
    for dets in ci_vectors:
        for key in dets:
            alldets.add(key)
    ndets = len(alldets)
    nstates = len(ci_vectors)
    norb = len(next(iter(alldets)))

    string = '%i %i %i\n' % (nstates, norb, ndets)
    for det in sorted(alldets, reverse=True):
        for o in det:
            if o == 0:
                string += 'e'
            elif o == 1:
                string += 'a'
            elif o == 2:
                string += 'b'
            elif o == 3:
                string += 'd'
        for istate in range(len(ci_vectors)):
            if det in ci_vectors[istate]:
                string += ' %11.7f ' % ci_vectors[istate][det]
            else:
                string += ' %11.7f ' % 0.
        string += '\n'
    return string

# ======================================================================= #


def saveAOmatrix(WORKDIR, QMin):
    # filename=os.path.join(WORKDIR,'ORCA.log')
    # NAO,Smat=get_smat(filename)
    # filename=os.path.join(WORKDIR,'ORCA.molden.input')
    # NAO,Smat=get_smat_from_Molden(filename)
    filename = os.path.join(WORKDIR, 'ORCA.gbw')
    NAO, Smat = get_smat_from_gbw(filename)

    string = '%i %i\n' % (NAO, NAO)
    for irow in range(NAO):
        for icol in range(NAO):
            string += '% .7e ' % (Smat[icol][irow])
        string += '\n'
    filename = os.path.join(QMin['savedir'], 'AO_overl')
    writefile(filename, string)
    if PRINT:
        print(shorten_DIR(filename))

# ======================================================================= #
# def get_smat(filename):

    # data=readfile(filename)

    # find MO block
    # iline=-1
    # NAO=0
    # while True:
        # iline+=1
        # if len(data)<=iline:
        # print('MOs not found!')
        # sys.exit(87)
        # line=data[iline]
        # if '# of contracted basis functions' in line:
        # NAO=int(line.split()[-1])
        # if 'OVERLAP MATRIX' in line:
        # break
    # if NAO==0:
        # print('Number of basis functions not found!')
        # sys.exit(88)
    # iline+=2

    # read matrix
    # nblock=6
    # ao_ovl=[ [ 0. for i in range(NAO) ] for j in range(NAO) ]
    # for i in range(NAO):
        # for j in range(NAO):
        # jline=iline + (i/nblock)*(NAO+1)+1+j
        # jcol =1+i%nblock
        # ao_ovl[i][j]=float(data[jline].split()[jcol])

    # return NAO,ao_ovl

# ======================================================================= #


def get_smat_from_gbw(file1, file2=''):

    if not file2:
        file2 = file1

    # run orca_fragovl
    string = 'orca_fragovl %s %s' % (file1, file2)
    try:
        proc = sp.Popen(string, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(89)
    comm = proc.communicate()[0].decode()
    out = comm.split('\n')

    # get size of matrix
    for line in reversed(out):
        # print line
        s = line.split()
        if len(s) >= 1:
            NAO = int(line.split()[0]) + 1
            break

    # read matrix
    nblock = 6
    ao_ovl = [[0. for i in range(NAO)] for j in range(NAO)]
    for x in range(NAO):
        for y in range(NAO):
            block = x // nblock
            xoffset = x % nblock + 1
            yoffset = block * (NAO + 1) + y + 10
            ao_ovl[x][y] = float(out[yoffset].split()[xoffset])

    return NAO, ao_ovl


# ======================================================================= #
# def get_smat_from_Molden(file1, file2=''):

    # read file1
    # molecule=read_molden(file1)

    # read file2:
    # if file2:
    # molecule.extend(read_molden(file2))
    # pprint.pprint(molecule)

    # make PyQuante object
    # try:
    # from PyQuante.Ints import getS
    # from PyQuante.Basis.basis import BasisSet
    # from PyQuante.CGBF import CGBF
    # from PyQuante.PGBF import PGBF
    # from PyQuante import Molecule
    # from PyQuante.shell import Shell
    # except ImportError:
    # print('Could not import PyQuante!')
    # sys.exit(90)

    # class moldenBasisSet(BasisSet):
    # def __init__(self, molecule):
    # sym2powerlist = {
    # 'S' : [(0,0,0)],
    # 'P' : [(1,0,0),(0,1,0),(0,0,1)],
    # 'D' : [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)],
    # 'F' : [(3,0,0),(0,3,0),(0,0,3),(1,2,0),(2,1,0),(2,0,1),
    # (1,0,2),(0,1,2),(0,2,1), (1,1,1)],
    # 'G' : [(4,0,0),(0,4,0),(0,0,4),(3,1,0),(3,0,1),(1,3,0),
    # (0,3,1),(1,0,3),(0,1,3),(2,2,0),(2,0,2),(0,2,2),
    # (2,1,1),(1,2,1),(1,1,2)]
    # }


    # make molecule
    # atomlist=[]
    # for atom in molecule:
    # atomlist.append( (atom['el'], tuple(atom['coord'])) )
    # target=Molecule('default',atomlist,units='bohr')

    # self.bfs=[]
    # self.shells=[]
    # for iatom,atom in enumerate(molecule):
    # for bas in atom['basis']:
    # shell=Shell(bas[0])
    # for power in sym2powerlist[bas[0]]:
    # cgbf = CGBF(target[iatom].pos(), power, target[iatom].atid)
    # for alpha, coef in bas[1:]:
    # angular=sum(power)
    # coef*=alpha**(-(0.75+angular*0.5))  *  2**(-angular)  *  (2.0/math.pi)**(-0.75)
    # cgbf.add_primitive(alpha,coef)
    # cgbf.normalize()
    # self.bfs.append(cgbf)
    # shell.append(cgbf, len(self.bfs)-1)
    # self.shells.append(shell)

    # a=moldenBasisSet(molecule)
    # pprint.pprint(a.__dict__)

    # S=getS(a)
    # S=S.tolist()
    # return len(S),S


# ======================================================================= #
def read_molden(filename):
    data = readfile(filename)

    molecule = []
    # get geometry
    for iline, line in enumerate(data):
        if '[atoms]' in line.lower():
            break
    else:
        print('No geometry found in %s!' % (filename))
        sys.exit(91)

    if 'au' in line.lower():
        factor = 1.
    elif 'angstrom' in line.lower():
        factor = au2a

    while True:
        iline += 1
        line = data[iline]
        if '[' in line:
            break
        s = line.lower().split()
        atom = {'el': s[0], 'coord': [float(i) * factor for i in s[3:6]], 'basis': []}
        molecule.append(atom)

    # get basis set
    for iline, line in enumerate(data):
        if '[gto]' in line.lower():
            break
    else:
        print('No geometry found in %s!' % (filename))
        sys.exit(92)

    shells = {'s': 1, 'p': 3, 'd': 6, 'f': 10, 'g': 15}
    while True:
        iline += 1
        line = data[iline]
        if '[' in line:
            break
        s = line.lower().split()
        if len(s) == 0:
            continue
        if not s[0] in shells:
            iatom = int(s[0]) - 1
        else:
            newbf = [s[0].upper()]
            nprim = int(s[1])
            for iprim in range(nprim):
                iline += 1
                s = data[iline].split()
                newbf.append((float(s[0]), float(s[1])))
            molecule[iatom]['basis'].append(newbf)
    return molecule


# ======================================================================= #
def mkdir(DIR):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR):
            print('%s exists and is a file!' % (DIR))
            sys.exit(93)
        elif os.path.isdir(DIR):
            if DEBUG:
                print('Remake\t%s' % DIR)
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            if DEBUG:
                print('Make\t%s' % DIR)
            os.makedirs(DIR)
        except OSError:
            print('Can not create %s\n' % (DIR))
            sys.exit(94)

# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH) and crucial:
        print('Source %s does not exist, cannot create link!' % (PATH))
        sys.exit(95)
    if os.path.islink(NAME):
        if not os.path.exists(NAME):
            # NAME is a broken link, remove it so that a new link can be made
            os.remove(NAME)
        else:
            # NAME is a symlink pointing to a valid file
            if force:
                # remove the link if forced to
                os.remove(NAME)
            else:
                print('%s exists, cannot create a link of the same name!' % (NAME))
                if crucial:
                    sys.exit(96)
                else:
                    return
    elif os.path.exists(NAME):
        # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
        print('%s exists, cannot create a link of the same name!' % (NAME))
        if crucial:
            sys.exit(97)
        else:
            return
    os.symlink(PATH, NAME)

# =============================================================================================== #
# =============================================================================================== #
# =======================================  TheoDORE ============================================= #
# =============================================================================================== #
# =============================================================================================== #


def run_theodore(QMin, errorcodes):

    if 'theodore' in QMin:
        print('>>>>>>>>>>>>> Starting the TheoDORE job execution')

        for ijob in QMin['jobs']:
            if not QMin['jobs'][ijob]['restr']:
                if DEBUG:
                    print('Skipping Job %s because it is unrestricted.' % (ijob))
                continue
            else:
                mults = QMin['jobs'][ijob]['mults']
                gsmult = mults[0]
                ns = 0
                for i in mults:
                    ns += QMin['states'][i - 1] - (i == gsmult)
                if ns == 0:
                    if DEBUG:
                        print('Skipping Job %s because it contains no excited states.' % (ijob))
                    continue
            WORKDIR = os.path.join(QMin['scratchdir'], 'master_%i' % ijob)
            setupWORKDIR_TH(WORKDIR, QMin)
            os.environ
            errorcodes['theodore_%i' % ijob] = runTHEODORE(WORKDIR, QMin['theodir'])

        # Error code handling
        j = 0
        string = 'Error Codes:\n'
        for i in errorcodes:
            if 'theodore' in i:
                string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
                j += 1
                if j == 4:
                    j = 0
                    string += '\n'
        print(string)
        if any((i != 0 for i in errorcodes.values())):
            print('Some subprocesses did not finish successfully!')
            sys.exit(98)

        print('')

    return errorcodes

# ======================================================================= #


def setupWORKDIR_TH(WORKDIR, QMin):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

    # write dens_ana.in
    inputstring = '''rtype='cclib'
rfile='ORCA.log'
read_binary=True
jmol_orbitals=False
molden_orbitals=False
Om_formula=2
eh_pop=1
comp_ntos=True
print_OmFrag=True
output_file='tden_summ.txt'
prop_list=%s
at_lists=%s
''' % (str(QMin['template']['theodore_prop']), str(QMin['template']['theodore_fragment']))

    filename = os.path.join(WORKDIR, 'dens_ana.in')
    writefile(filename, inputstring)
    fromfile = os.path.join(WORKDIR, 'ORCA.cis')
    tofile = os.path.join(WORKDIR, 'orca.cis')
    link(fromfile, tofile)
    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(inputstring)
        print('TheoDORE input written to: %s' % (filename))
        print('====================================================================')

    return

# ======================================================================= #


def runTHEODORE(WORKDIR, THEODIR):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = os.path.join(THEODIR, 'bin', 'analyze_tden.py')
    stdoutfile = open(os.path.join(WORKDIR, 'theodore.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'theodore.err'), 'w')
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(99)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    return runerror

# =============================================================================================== #
# =============================================================================================== #
# =======================================  Dyson and overlap calcs ============================== #
# =============================================================================================== #
# =============================================================================================== #


def run_wfoverlap(QMin, errorcodes):

    print('>>>>>>>>>>>>> Starting the WFOVERLAP job execution')

    # do Dyson calculations
    if 'ion' in QMin:
        for ionpair in QMin['ionmap']:
            WORKDIR = os.path.join(QMin['scratchdir'], 'Dyson_%i_%i_%i_%i' % ionpair)
            files = {'aoovl': 'AO_overl',
                     'det.a': 'dets.%i' % ionpair[0],
                     'det.b': 'dets.%i' % ionpair[2],
                     'mo.a': 'mos.%i' % ionpair[1],
                     'mo.b': 'mos.%i' % ionpair[3]}
            setupWORKDIR_WF(WORKDIR, QMin, files)
            errorcodes['Dyson_%i_%i_%i_%i' % ionpair] = runWFOVERLAP(WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

    # do overlap calculations
    if 'overlap' in QMin:
        get_Double_AOovl_gbw(QMin)
        for m in itmult(QMin['states']):
            job = QMin['multmap'][m]
            WORKDIR = os.path.join(QMin['scratchdir'], 'WFOVL_%i_%i' % (m, job))
            files = {'aoovl': 'AO_overl.mixed',
                     'det.a': 'dets.%i.old' % m,
                     'det.b': 'dets.%i' % m,
                     'mo.a': 'mos.%i.old' % job,
                     'mo.b': 'mos.%i' % job}
            setupWORKDIR_WF(WORKDIR, QMin, files)
            errorcodes['WFOVL_%i_%i' % (m, job)] = runWFOVERLAP(WORKDIR, QMin['wfoverlap'], memory=QMin['memory'], ncpu=QMin['ncpu'])

    # Error code handling
    j = 0
    string = 'Error Codes:\n'
    for i in errorcodes:
        if 'Dyson' in i or 'WFOVL' in i:
            string += '\t%s\t%i' % (i + ' ' * (10 - len(i)), errorcodes[i])
            j += 1
            if j == 4:
                j = 0
                string += '\n'
    print(string)
    if any((i != 0 for i in errorcodes.values())):
        print('Some subprocesses did not finish successfully!')
        sys.exit(100)

    print('')

    return errorcodes

# ======================================================================= #


def setupWORKDIR_WF(WORKDIR, QMin, files):
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary files from pwd and savedir

    # setup the directory
    mkdir(WORKDIR)

    # write wfovl.inp
    inputstring = '''mix_aoovl=aoovl
a_mo=mo.a
b_mo=mo.b
a_det=det.a
b_det=det.b
a_mo_read=0
b_mo_read=0
ao_read=0
'''
    if 'ion' in QMin:
        if QMin['ndocc'] > 0:
            inputstring += 'ndocc=%i\n' % (QMin['ndocc'])
    if QMin['ncpu'] >= 8:
        inputstring += 'force_direct_dets\n'
    filename = os.path.join(WORKDIR, 'wfovl.inp')
    writefile(filename, inputstring)
    if DEBUG:
        print('================== DEBUG input file for WORKDIR %s =================' % (shorten_DIR(WORKDIR)))
        print(inputstring)
        print('wfoverlap input written to: %s' % (filename))
        print('====================================================================')

    # link input files from save
    linkfiles = ['aoovl', 'det.a', 'det.b', 'mo.a', 'mo.b']
    for f in linkfiles:
        fromfile = os.path.join(QMin['savedir'], files[f])
        tofile = os.path.join(WORKDIR, f)
        link(fromfile, tofile)

    return

# ======================================================================= #


def runWFOVERLAP(WORKDIR, WFOVERLAP, memory=100, ncpu=1):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = WFOVERLAP + ' -m %i' % (memory) + ' -f wfovl.inp'
    stdoutfile = open(os.path.join(WORKDIR, 'wfovl.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'wfovl.err'), 'w')
    os.environ['OMP_NUM_THREADS'] = str(ncpu)
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (shorten_DIR(WORKDIR), starttime, shorten_DIR(string)))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(101)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write('FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (shorten_DIR(WORKDIR), endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    return runerror


# ======================================================================= #
def get_Double_AOovl_gbw(QMin):

    # get geometries
    # filename1=os.path.join(QMin['savedir'],'ORCA.molden.1.old')
    # filename2=os.path.join(QMin['savedir'],'ORCA.molden.1')
    job = sorted(QMin['jobs'].keys())[0]
    filename1 = os.path.join(QMin['savedir'], 'ORCA.gbw.%i.old' % job)
    filename2 = os.path.join(QMin['savedir'], 'ORCA.gbw.%i' % job)

    #
    # NAO,Smat=get_smat_from_Molden(filename1,filename2)
    NAO, Smat = get_smat_from_gbw(filename1, filename2)

    # Smat is now full matrix NAO*NAO
    # we want the lower left quarter, but transposed
    # string='%i %i\n' % (NAO/2,NAO/2)
    # for irow in range(NAO/2,NAO):
    # for icol in range(0,NAO/2):
    # string+='% .15e ' % (Smat[icol][irow])          # note the exchanged indices => transposition
    # string+='\n'

    # Smat is already off-diagonal block matrix NAO*NAO
    # we want the lower left quarter, but transposed
    string = '%i %i\n' % (NAO, NAO)
    for irow in range(0, NAO):
        for icol in range(0, NAO):
            string += '% .15e ' % (Smat[irow][icol])          # note the exchanged indices => transposition
        string += '\n'
    filename = os.path.join(QMin['savedir'], 'AO_overl.mixed')
    writefile(filename, string)
    return













# =============================================================================================== #
# =============================================================================================== #
# ====================================== DFTB output parsing ================================ #
# =============================================================================================== #
# =============================================================================================== #
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
                sys.exit(100)
            if dimF != len(dimA):
                print('Error in Dimension of Tag %s' % (prop))
                sys.exit(101)

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


def readAutoTag(filename,properties):
    data = {}
    file = readfile(filename)
    for prop in properties:
        result = getSingleTag(prop,file)
        if len(result) != 0:
            data[prop] = result

    return data

def getCoupleNacvs(nacvs,QMin):
    nmstates = QMin['nmstates']
    totalCoupleStates = int(nmstates*(nmstates-1)/2)
    WORKDIR = QMin['savedir']
    if totalCoupleStates != nacvs.shape[0]:
        print("Total number of couple states in non-adiabatic coupling vectors does not match with the outputs of DFTB+")
        sys.exit(101)

    filename = os.path.join(WORKDIR,'nacvsOld.dat')
    phase = np.ones(totalCoupleStates)
    # Step 0, we save the nacvs without correction
    if int(QMin['step'][0]) == 0:
        print('Saving NACVs in the zero step. Not phase correction here...')
    else:
        # Report error if we can not find old nacs in the step >= 1
        if not os.path.isfile(filename):
            print('File %s not found, cannot perform Overlap!' % (filename))
            sys.exit(77)
        # Calculate the correct phase of nacv
        else:
            print('Loading old nacvs with correct phase. Making the phase correction to the new NACVs.')
            nacvsOld = np.loadtxt(filename)
            # Check dimensions of old nacv
            if nacvsOld.shape[0] != totalCoupleStates:
                print("Total number of couple states in Old non-adiabatic coupling vectors does not match with current one.")
                sys.exit(101)
            # Make overlap between current an old nacvs
            for ii in range(totalCoupleStates):
                dot = np.dot(nacvsOld[ii],nacvs[ii].reshape(-1))
                if dot < 0.:
                    phase[ii] = -1.

    # Save nacvs in QMout with phase corrected
    coupleNacvs = {}
    count = 0
    for ii in range(nmstates):
        for jj in range(ii+1,nmstates):
            nacvs[count] *= phase[count]
            coupleNacvs[(ii+1,jj+1)] = nacvs[count]
            count += 1

    # Save current nacvs for the next step with the phase corrected
    xnac = nacvs.reshape((totalCoupleStates,-1))
    np.savetxt(filename,xnac)
    return coupleNacvs

def readCIvectors(filename):
    if not os.path.isfile(filename):
       print('File %s not found, cannot perform Overlap!' % (filename))
       sys.exit(77)

    ff = open(filename,"r")
    file = ff.readlines()
    ff.close()

    ncielem = int(file[0].split()[0])
    nstates = int(file[0].split()[1])
    civects = np.zeros((nstates,ncielem))
    how_many_lines = ncielem // 6 + (0 if ncielem % 6 == 0 else 1)

    for ii in range(1,len(file)):
        iline = file[ii]
        if 'S' in iline:
            istate = int(iline.split()[0]) - 1
            ciarr = []
            for kk in range(ii+1,ii+how_many_lines+1):
                kline = file[kk].split()
                for ele in kline:
                    ciarr.append(float(ele))
            if len(ciarr) != ncielem:
                print('Dimension error in CIvector of the state %i for file %s!' % (istate,filename))
                sys.exit(77)
            for kk in range(0,ncielem):
                civects[istate,kk] = ciarr[kk]

            norm = np.dot(civects[istate,:],civects[istate,:])
            civects[istate,:] = civects[istate,:] / np.sqrt(norm)

    return civects

def getOverlap(QMin):
    # GDM: TODO: this routine is obsolete, now we are performed the phase correction
    #            directly on the NACs.
    WORKDIR = QMin['savedir']

    # Read New/Current CI vectors
    filename = os.path.join(WORKDIR,'XplusY.DAT.1')
    cinew = readCIvectors(filename)

    # Read Old CI vectors
    filename = os.path.join(WORKDIR,'XplusY.DAT.1.old')
    ciold = readCIvectors(filename)

    # Read phases of old wavefunction
    """
    nmstates = QMin['nmstates']
    istep    = int(QMin['step'][0])
    filename = os.path.join(WORKDIR,'phasesOld.dat')
    if not os.path.isfile(filename) and istep > 1:
       print('File %s not found, cannot perform Overlap!' % (filename))
       sys.exit(77)
    elif not os.path.isfile(filename):
        phase = np.ones(nmstates,dtype=np.complex_)
    else:
        phase = np.loadtxt(filename, dtype=np.complex_)

    # Make overlap
    ovlp = makecmatrix(nmstates, nmstates)
    ovlp[0][0] = complex(1., 0.)
    for ii in range(1,nmstates):
        for jj in range(1,ii+1):
            # GDM: TODO: NO FUNCA, deje la correcta
            dot = np.dot(cinew[ii-1,:],phase[jj].real*ciold[jj-1,:])
            ovlp[ii][jj] = complex(dot,0.)
            ovlp[jj][ii] = complex(dot,0.)
    """
    nmstates = QMin['nmstates']
    ovlp = makecmatrix(nmstates, nmstates)
    for ii in range(1,nmstates):
        ovlp[ii][ii] = complex(1., 0.)
    return ovlp

def getQMout(QMin):

    if PRINT:
        print('>>>>>>>>>>>>> Reading output files')
    starttime = datetime.datetime.now()

    QMout = {}
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    joblist = QMin['joblist']
    gradmap = QMin['gradmap']
    nacmap = QMin['nacmap']

    # Read all the outputs of dftb
    dftbResults = {}
    for job in joblist:

        # Read master calculation
        WORKDIR = os.path.join(QMin['scratchdir'], 'master_%i' % (job))
        tagfile = os.path.join(WORKDIR, 'autotest_master.tag')
        properties = ['exc_energies_sqr','exc_oscillator','mermin_energy']
        dftbResults['master_%i' % (job) ] = readAutoTag(tagfile,properties)

        # Read Gradients of all states
        for grad in gradmap:
            # Read Excitate states properties
            properties = ['forces','dipole_moments','mermin_energy']
            tagfile = os.path.join(WORKDIR, 'autotest_grad_%i.tag' % (grad[1]))
            dftbResults['grad_%i' % (grad[1])] = readAutoTag(tagfile,properties)

        # Read Non-adiabatic coupling vectores
        if nacmap:
            properties = ['nac_vectors']
            tagfile = os.path.join(WORKDIR, 'autotest_nacvs.tag')
            scratchNacvs = readAutoTag(tagfile,properties)
            dftbResults['nacvs'] = getCoupleNacvs(scratchNacvs['nac_vectors'],QMin)

    # Hamiltonian
    if 'h' in QMin:
        # make Hamiltonian
        if 'h' not in QMout:
            QMout['h'] = makecmatrix(nmstates, nmstates)
        # go through all jobs
        for job in joblist:
            energies = getenergy(dftbResults['master_%i' % (job)], job, QMin)
            mults = QMin['multmap'][-job]
            for i in range(nmstates):
                for j in range(nmstates):
                    m1, s1, ms1 = tuple(QMin['statemap'][i + 1])
                    m2, s2, ms2 = tuple(QMin['statemap'][j + 1])
                    if m1 not in mults or m2 not in mults:
                        continue
                    if i == j:
                        QMout['h'][i][j] = energies[(m1, s1)]

    # GDM: TODO: here they put the dipole moments and also the transition dipoles
    if 'dm' in QMin:
       QMout['dm'] = [makecmatrix(nmstates, nmstates) for i in range(3)]

    # GDM: DFTB+ prints out the forces, not gradients.
    if 'grad' in QMin:
        QMout['grad'] = [[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)]
        for grad in QMin['gradmap']:
            istate = grad[1]
            gg = dftbResults['grad_%i' % (istate)]['forces']

            # Save the gradients
            for iatom in range(natom):
                for xyz in range(3):
                    QMout['grad'][istate - 1][iatom][xyz] = gg[iatom,xyz] * (-1.)

        # Negleted Gradient
        if QMin['neglected_gradient'] != 'zero':
            print('Neglected gradient is not allowed with DFTB+ for the moment...')
            sys.exit(33)


    # Overlap and phases
    if 'overlap' in QMin:
        QMout['overlap'] = getOverlap(QMin)

    if 'phases' in QMin:
        QMout['phases'] = [complex(1., 0.) for i in range(nmstates)]

        # GDM: TODO: For the moment the only phase correction is performed on NACs
        #if 'overlap' in QMout:
            # 
            #for ii in range(1,nmstates):
            #    if QMout['overlap'][ii][ii].real < 0.:
            #        QMout['phases'][ii] = complex(-1., 0.)
            #filename = os.path.join(QMin['savedir'],'phasesOld.dat')
            #print('Saving phases in %s' % (shorten_DIR(filename)))
            #np.savetxt(filename,QMout['phases'])

    # Non adiabatic couplings
    if 'nacdr' in QMin:
        QMout['nacdr'] = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)] for l in range(nmstates)]
        for nac in QMin['nacmap']:
            # TODO: just for checking but nacmap always has ele 3 > ele 1
            if nac[3] <= nac[1]:
                print("Error in nacmap elements")
                sys.exit(100)
            gg = dftbResults['nacvs'][(nac[1],nac[3])]

            # DEBUG #####################################
            # GDM: TODO: just to check the norm of NACV
            norm = 0.
            for iatom in range(natom):
                for ixyz in range(3):
                    norm += gg[iatom,ixyz] * gg[iatom,ixyz]
            Egap = QMout['h'][nac[3]-1][nac[3]-1] - QMout['h'][nac[1]-1][nac[1]-1]
            print('Norm NACV and Energy Gap S%i-S%i: %f %f' % (nac[1]-1,nac[3]-1,np.sqrt(norm),Egap))
            #############################################

            ggpos = makermatrix(3,natom)
            ggneg = makermatrix(3,natom)
            for iatom in range(natom):
                for ixyz in range(3):
                    ggpos[iatom][ixyz] = gg[iatom,ixyz] 
                    ggneg[iatom][ixyz] = gg[iatom,ixyz] * -1.
            for istate in QMin['statemap']:
                for jstate in QMin['statemap']:
                    state1 = QMin['statemap'][istate]
                    state2 = QMin['statemap'][jstate]
                    if not state1[2] == state2[2]:
                        continue
                    if (state1[0], state1[1], state2[0], state2[1]) == nac:
                        QMout['nacdr'][istate-1][jstate-1] = ggpos
                    elif (state2[0], state2[1], state1[0], state1[1]) == nac:
                        QMout['nacdr'][istate-1][jstate-1] = ggneg

    endtime = datetime.datetime.now()
    if PRINT:
        print("Readout Runtime: %s" % (endtime - starttime))

    if DEBUG:
        print("GDM: For the moment there is no debug implementation... Should be soon!.")

    # Saving important files
    if QMin['save_stuff']:
        copydir = os.path.join(QMin['savedir'], 'save_stuff')
        if not os.path.isdir(copydir):
            mkdir(copydir)
        for job in joblist:
            # GDM: For the moment we are saving common files, gs results and common exc states
            #      TODO: probably we will need to put more here
            outfile = os.path.join(QMin['scratchdir'], 'master_%i/autotest_master.tag' % (job))
            shutil.copy(outfile, os.path.join(copydir, 'autotest_master.tag'))

            outfile = os.path.join(QMin['scratchdir'], 'master_%i/autotest_nacvs.tag' % (job))
            shutil.copy(outfile, os.path.join(copydir, 'autotest_nacvs.tag'))

    return QMin, QMout

# ======================================================================= #


def getenergy(dftbr, ijob, QMin):
    # ground state energy in Hartree
    gsenergy = dftbr['mermin_energy'][0]

    # figure out the excited state settings
    mults = QMin['jobs'][ijob]['mults']
    restr = QMin['jobs'][ijob]['restr']
    gsmult = mults[0]
    estates_to_extract = deepcopy(QMin['states'])
    estates_to_extract[gsmult - 1] -= 1
    for imult in range(len(estates_to_extract)):
        if not imult + 1 in mults:
            estates_to_extract[imult] = 0
    for imult in range(len(estates_to_extract)):
        if imult + 1 in mults:
            estates_to_extract[imult] = max(estates_to_extract)

    # Get excitation energies
    energies = {(gsmult, 1): gsenergy}
    for imult in mults:
        nstates = estates_to_extract[imult - 1]
        for ii in range(nstates):
            e = gsenergy+np.sqrt(dftbr['exc_energies_sqr'][ii])
            energies[(imult, ii + 1 + (gsmult == imult))] = e
    return energies

## ======================================================================= #


def getsocm(outfile, ijob, QMin):

    # read the standard out into memory
    out = readfile(outfile)
    if PRINT:
        print('SOC:      ' + shorten_DIR(outfile))



    # get number of states (nsing=ntrip in Orca)
    for line in out:
        if 'Number of roots to be determined' in line:
            nst = int(line.split()[-1])
            break
    nrS = nst
    nrT = nst

    # make statemap for the state ordering of the SO matrix
    inv_statemap = {}
    inv_statemap[(1, 1, 0.0)] = 1
    i = 1
    for x in range(nrS):
        i += 1
        inv_statemap[(1, x + 2, 0.0)] = i
    spin = [0.0, -1.0, +1.0]
    for y in range(3):
        for x in range(nrT):
            i += 1
            inv_statemap[(3, x + 1, spin[y])] = i
    #pprint.pprint( inv_statemap)

    # get matrix
    iline = -1
    while True:
        iline += 1
        line = out[iline]
        if 'The full SOC matrix' in line:
            break
    iline += 5
    ncol = 6
    real = [[0 + 0j for i in range(4 * nst + 1)] for j in range(4 * nst + 1)]
    for x in range(len(real)):
        for y in range(len(real[0])):
            block = x // ncol
            xoffset = 1 + x % ncol
            yoffset = block * (4 * nst + 2) + y
            # print iline,x,y,block,xoffset,yoffset
            val = float(out[iline + yoffset].split()[xoffset])
            if abs(val) > 1e-16:
                real[y][x] = val

    iline += ((4 * nst) // ncol + 1) * (4 * nst + 2) + 2
    for x in range(len(real)):
        for y in range(len(real[0])):
            block = x // ncol
            xoffset = 1 + x % ncol
            yoffset = block * (4 * nst + 2) + y
            val = float(out[iline + yoffset].split()[xoffset])
            if abs(val) > 1e-16:
                real[y][x] += (0 + 1j) * val

    #pprint.pprint( real)


    return real, inv_statemap

# ======================================================================= #


def gettdm(logfile, ijob, QMin):

    # open file
    f = readfile(logfile)
    if PRINT:
        print('Dipoles:  ' + shorten_DIR(logfile))

    # figure out the excited state settings
    mults = QMin['jobs'][ijob]['mults']
    if 3 in mults and QMin['OrcaVersion'] < (4, 1):
        mults = [3]
    restr = QMin['jobs'][ijob]['restr']
    gsmult = mults[0]
    estates_to_extract = deepcopy(QMin['states'])
    estates_to_extract[gsmult - 1] -= 1
    for imult in range(len(estates_to_extract)):
        if not imult + 1 in mults:
            estates_to_extract[imult] = 0

    # print "getting cool dipoles"
    # extract transition dipole moments
    dipoles = {}
    for imult in mults:
        if not imult == gsmult:
            continue
        nstates = estates_to_extract[imult - 1]
        if nstates > 0:
            for iline, line in enumerate(f):
                if '  ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
                    # print line
                    for istate in range(nstates):
                        shift = 5 + istate
                        s = f[iline + shift].split()
                        dm = [float(i) for i in s[5:8]]
                        dipoles[(imult, istate + 1 + (gsmult == imult))] = dm
    # print dipoles
    return dipoles

# ======================================================================= #


def getdm(logfile, isgs):

    # open file
    f = readfile(logfile)
    if PRINT:
        print('Dipoles:  ' + shorten_DIR(logfile))

    if isgs:
        findstring = 'ORCA ELECTRIC PROPERTIES CALCULATION'
    else:
        findstring = '*** CIS RELAXED DENSITY ***'
    for iline, line in enumerate(f):
        if findstring in line:
            break
    while True:
        iline += 1
        line = f[iline]
        if 'Total Dipole Moment' in line:
            s = line.split()
            dmx = float(s[4])
            dmy = float(s[5])
            dmz = float(s[6])
            dm = [dmx, dmy, dmz]
            return dm

# ======================================================================= #


def getgrad(logfile, QMin):

    # initialize
    natom = QMin['natom']
    g = [[0. for i in range(3)] for j in range(natom)]

    # read file
    if os.path.isfile(logfile):
        out = readfile(logfile)
        if PRINT:
            print('Gradient: ' + shorten_DIR(logfile))

        # get gradient
        string = 'The current gradient in Eh/bohr'
        shift = 2
        for iline, line in enumerate(out):
            if string in line:
                for iatom in range(natom):
                    for ixyz in range(3):
                        s = out[iline + shift + 3 * iatom + ixyz]
                        g[iatom][ixyz] = float(s)

    # read binary file otherwise
    else:
        logfile += '.grad.tmp'
        Gfile = open(logfile, 'rb')
        if PRINT:
            print('Gradient: ' + shorten_DIR(logfile))

        # get gradient
        Gfile.read(8 + 28 * natom)    # skip header
        for iatom in range(natom):
            for ixyz in range(3):
                f = struct.unpack('d', Gfile.read(8))[0]
                g[iatom][ixyz] = f

    return g

# ======================================================================= #


def getgrad_from_log(logfile, QMin):

    # read file
    out = readfile(logfile)
    if PRINT:
        print('Gradient: ' + shorten_DIR(logfile))

    # initialize
    natom = QMin['natom']
    g = [[0. for i in range(3)] for j in range(natom)]

    # find gradients
    iline = -1
    while True:
        iline += 1
        line = out[iline]
        if 'ORCA SCF GRADIENT CALCULATION' in line:
            break


    return g

# ======================================================================= #


def getpcgrad(logfile, QMin):

    # read file
    out = readfile(logfile)
    if PRINT:
        print('Gradient: ' + shorten_DIR(logfile))

    # initialize
    # natom=len(QMin['pointcharges'])
    # g=[ [ 0. for i in range(3) ] for j in range(natom) ]

    # get gradient
    # for iatom in range(natom):
    #     for ixyz in range(3):
    #       s=out[iatom+1].split()
    #       g[iatom][ixyz]=float(s[ixyz])
    # return g
    g = []
    for iatom in range(len(out) - 1):
        atom_grad = [0. for i in range(3)]
        s = out[iatom + 1].split()
        for ixyz in range(3):
            atom_grad[ixyz] = float(s[ixyz])
        g.append(atom_grad)
    return g

## ======================================================================= #
# def get_qmmm_energies(outfile,coupling):

    # out=readfile(outfile)
    # if PRINT:
    # print('QMMM:     '+shorten_DIR(outfile))

    # if coupling==1:
    #startstring='Q M / M M      E N E R G Y'
    # shift=2
    # elif coupling==2:
    #startstring='These results include the electrostatic interaction between QM and MM systems'
    # shift=0

    # toextract={'bond_mm':     (5,2),
    # 'angle_mm':    (6,2),
    # 'torsion_mm':  (7,2),
    # 'VdW_mm':      (10,4),
    # 'elstat_mm':   (11,2),
    # 'VdW_qmmm':    (10,5),
    # 'elstat_qmmm': (11,3)
    # }

    # iline=-1
    # while True:
    # iline+=1
    # if startstring in out[iline]:
    # break
    # iline+=shift

    # energies={}
    # for label in toextract:
    # t=toextract[label]
    # e=float(out[iline+t[0]].split()[t[1]])
    # energies[label]=e

    # return energies

# ======================================================================= #


def getsmate(out, s1, s2):
    ilines = -1
    while True:
        ilines += 1
        if ilines == len(out):
            print('Overlap of states %i - %i not found!' % (s1, s2))
            sys.exit(103)
        if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
            break
    ilines += 1 + s1
    f = out[ilines].split()
    return float(f[s2 + 1])

# ======================================================================= #


def getDyson(out, s1, s2):
    ilines = -1
    while True:
        ilines += 1
        if ilines == len(out):
            print('Dyson norm of states %i - %i not found!' % (s1, s2))
            sys.exit(104)
        if containsstring('Dyson norm matrix <PsiA_i|PsiB_j>', out[ilines]):
            break
    ilines += 1 + s1
    f = out[ilines].split()
    return float(f[s2 + 1])

# ======================================================================= #


def get_theodore(sumfile, omffile, QMin):
    out = readfile(sumfile)
    if PRINT:
        print('TheoDORE: ' + shorten_DIR(sumfile))
    props = {}
    for line in out[2:]:
        s = line.replace('(', ' ').replace(')', ' ').split()
        if len(s) == 0:
            continue
        n = int(s[0])
        m = int(s[1])
        props[(m, n + (m == 1))] = [theo_float(i) for i in s[5:]]

    out = readfile(omffile)
    if PRINT:
        print('TheoDORE: ' + shorten_DIR(omffile))
    for line in out[1:]:
        s = line.replace('(', ' ').replace(')', ' ').split()
        if len(s) == 0:
            continue
        n = int(s[0])
        m = int(s[1])
        props[(m, n + (m == 1))].extend([theo_float(i) for i in s[4:]])

    return props

# ======================================================================= #


def theo_float(string):
    try:
        s = float(string)
    except ValueError:
        s = 0.
    return s

# =============================================================================================== #
# =============================================================================================== #
# ========================================= Miscellaneous ======================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def cleandir(directory):
    for data in os.listdir(directory):
        path = directory + '/' + data
        if os.path.isfile(path) or os.path.islink(path):
            if DEBUG:
                print('rm %s' % (path))
            try:
                os.remove(path)
            except OSError:
                print('Could not remove file from directory: %s' % (path))
        else:
            if DEBUG:
                print('')
            cleandir(path)
            os.rmdir(path)
            if DEBUG:
                print('rm %s' % (path))
    if PRINT:
        print('===> Cleaning up directory %s' % (directory))

# ======================================================================= #


def backupdata(backupdir, QMin):
    # save all files in savedir, except which have 'old' in their name
    ls = os.listdir(QMin['savedir'])
    for f in ls:
        ff = QMin['savedir'] + '/' + f
        if os.path.isfile(ff) and 'old' not in ff:
            step = int(QMin['step'][0])
            fdest = backupdir + '/' + f + '.stp' + str(step)
            shutil.copy(ff, fdest)

# =============================================================================================== #
# =============================================================================================== #
# ========================================= Main ================================================ #
# =============================================================================================== #
# =============================================================================================== #


def main():

    # Retrieve PRINT and DEBUG
    try:
        envPRINT = os.getenv('SH2Orc_PRINT')
        if envPRINT and envPRINT.lower() == 'false':
            global PRINT
            PRINT = False
        envDEBUG = os.getenv('SH2Orc_DEBUG')
        if envDEBUG and envDEBUG.lower() == 'true':
            global DEBUG
            DEBUG = True
    except ValueError:
        print('PRINT or DEBUG environment variables do not evaluate to numerical values!')
        sys.exit(105)

    # Process Command line arguments
    if len(sys.argv) != 2:
        print('Usage:\n./SHARC_DFTB.py <QMin>\n')
        print('version:', version)
        print('date:', versiondate)
        print('changelog:\n', changelogstring)
        sys.exit(106)
    QMinfilename = sys.argv[1]

    # Print header
    printheader()

    # Read QMinfile
    QMin = readQMin(QMinfilename)

    # get the job schedule
    QMin, schedule = generate_joblist(QMin)
    printQMin(QMin)
    if DEBUG:
        pprint.pprint(schedule, depth=3)

    # run all the ADF jobs
    errorcodes = runjobs(schedule, QMin)

    # read all the output files
    QMin, QMout = getQMout(QMin)
    if PRINT or DEBUG:
        printQMout(QMin, QMout)

    # backup data if requested
    if 'backup' in QMin:
        backupdata(QMin['backup'], QMin)

    # Measure time
    runtime = measuretime()
    QMout['runtime'] = runtime

    # Write QMout
    writeQMout(QMin, QMout, QMinfilename)

    # Remove Scratchfiles from SCRATCHDIR
    if not DEBUG:
        cleandir(QMin['scratchdir'])
        if 'cleanup' in QMin:
            cleandir(QMin['savedir'])

    print
    print(datetime.datetime.now())
    print('#================ END ================#')


if __name__ == '__main__':
    main()






# kate: indent-width 2
