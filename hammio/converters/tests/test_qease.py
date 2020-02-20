#!/usr/bin/env python
import numpy as np
from hammio.converters import yy_qease as yq

def example_scf(name):
  h4nosym_in = '''&CONTROL
   calculation     = 'scf'
   tprnfor         = .true.
   tstress         = .true.
   verbosity       = 'low'
   wf_collect      = .false.
/

&SYSTEM
   degauss         = 0.0063338
   ecutwfc         = 40
   ibrav           = 0
   input_dft       = 'vdw-df'
   nat             = 4
   nosym           = .true.
   ntyp            = 1
   occupations     = 'smearing'
   smearing        = 'f-d'
   tot_charge      = 0
/

&ELECTRONS
   conv_thr        = 1e-10
   diagonalization = 'david'
   electron_maxstep = 1000
   mixing_beta     = 0.7
   mixing_mode     = 'plain'
/


ATOMIC_SPECIES
   H  1.00794 H.pbe-kjpaw_psl.0.1.UPF

ATOMIC_POSITIONS bohr
   H  0.          0.          0.
   H  1.71532687  2.97102736  0.
   H  3.43064346  5.94205471  0.
   H  5.14597033  8.91308207  0.

K_POINTS automatic
   4 4 4  1 1 1

CELL_PARAMETERS bohr
        10.29201544       0.00000000       0.00000000
         0.00000000      11.88429868       0.00000000
         0.00000000       0.00000000      10.87537389'''

  # shortened output to save space
  h4nosym_out = '''
     Program PWSCF v.6.4 starts on 20Feb2020 at 13:28:37

     Parallel version (MPI), running on    16 processors

     MPI processes distributed on     1 nodes
     K-points division:     npool     =      16
     Waiting for input...
     Reading input from standard input

     Current dimensions of program PWSCF are:
     Max number of different atomic species (ntypx) = 10
     Max number of k-points (npk) =  40000
     Max angular momentum in pseudopotentials (lmaxx) =  3

     IMPORTANT: XC functional enforced from input :
     Exchange-correlation      = VDW-DF ( 1  4  4  0 1 0)
     Any further DFT definition will be discarded
     Please, verify this is what you really want

               file H.pbe-kjpaw_psl.0.1.UPF: wavefunction(s)  1S renormalized

     Subspace diagonalization in iterative solution of the eigenvalue problem:
     a serial algorithm will be used


     G-vector sticks info
     --------------------
     sticks:   dense  smooth     PW     G-vecs:    dense   smooth      PW
     Sum        1551    1551    433                45417    45417    6743



     bravais-lattice index     =            0
     lattice parameter (alat)  =      10.2920  a.u.
     unit-cell volume          =    1330.2038 (a.u.)^3
     number of atoms/cell      =            4
     number of atomic types    =            1
     number of electrons       =         4.00
     number of Kohn-Sham states=            6
     kinetic-energy cutoff     =      40.0000  Ry
     charge density cutoff     =     160.0000  Ry
     convergence threshold     =      1.0E-10
     mixing beta               =       0.7000
     number of iterations used =            8  plain     mixing
     Exchange-correlation      = VDW-DF ( 1  4  4  0 1 0)

     celldm(1)=  10.292015  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.154711   0.000000 )
               a(3) = (   0.000000   0.000000   1.056681 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  0.866018  0.000000 )
               b(3) = (  0.000000  0.000000  0.946360 )


     PseudoPot. # 1 for H  read from file:
     ./H.pbe-kjpaw_psl.0.1.UPF
     MD5 check sum: fbb168fef676c0aaadbafd73f5b1efa8
     Pseudo is Projector augmented-wave, Zval =  1.0
     Generated using "atomic" code by A. Dal Corso v.5.0.2 svn rev. 9415
     Shape of augmentation charge: PSQ
     Using radial grid of  929 points,  2 beta functions with:
                l(1) =   0
                l(2) =   0
     Q(r) pseudized with 0 coefficients


     vdW kernel table read from file vdW_kernel_table
     MD5 check sum: edb2d2c349e1b671170ac0b8bd431292

     atomic species   valence    mass     pseudopotential
        H              1.00     1.00794     H ( 1.00)

     No symmetry found



   Cartesian axes

     site n.     atom                  positions (alat units)
         1           H   tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           H   tau(   2) = (   0.1666658   0.2886730   0.0000000  )
         3           H   tau(   3) = (   0.3333306   0.5773461   0.0000000  )
         4           H   tau(   4) = (   0.4999964   0.8660191   0.0000000  )

     number of k points=    32  Fermi-Dirac smearing, width (Ry)=  0.0063
                       cart. coord. in units 2pi/alat
        k(    1) = (   0.1250000   0.1082522   0.1182950), wk =   0.0625000
        k(   32) = (   0.3750000  -0.1082522  -0.1182950), wk =   0.0625000

     Dense  grid:    45417 G-vectors     FFT dimensions: (  45,  48,  45)

     Estimated max dynamical RAM per process >      34.13 MB

     Estimated total dynamical RAM >     546.10 MB

     Initial potential from superposition of free atoms

     starting charge    3.99996, renormalised to    4.00000

     Starting wfcs are    4 randomized atomic wfcs +    2 random wfcs
     Checking if some PAW data can be deallocated...

     total cpu time spent up to now is        2.2 secs

     Self-consistent Calculation

     iteration #  1     ecut=    40.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  1.00E-02,  avg # of iterations =  7.0

     Threshold (ethr) on eigenvalues was too large:
     Diagonalizing with lowered threshold

     Davidson diagonalization with overlap
     ethr =  3.32E-04,  avg # of iterations =  3.0

     negative rho (up, down):  7.207E-07 0.000E+00

     total cpu time spent up to now is        4.0 secs

     total energy              =      -4.25515653 Ry
     Harris-Foulkes estimate   =      -4.25735807 Ry
     estimated scf accuracy    <       0.01311580 Ry

     iteration #  9     ecut=    40.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  4.69E-11,  avg # of iterations =  2.2

     negative rho (up, down):  1.048E-06 0.000E+00

     total cpu time spent up to now is       13.8 secs

     total energy              =      -4.25665619 Ry
     Harris-Foulkes estimate   =      -4.25665619 Ry
     estimated scf accuracy    <          2.1E-10 Ry

     iteration # 10     ecut=    40.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  5.26E-12,  avg # of iterations =  1.0

     negative rho (up, down):  1.048E-06 0.000E+00

     total cpu time spent up to now is       14.9 secs

     End of self-consistent calculation

          k = 0.1250 0.1083 0.1183 (  5675 PWs)   bands (ev):

    -8.5133  -7.0115  -5.2839  -3.0057  -0.4129   1.0347

          k = 0.3750-0.1083-0.1183 (  5677 PWs)   bands (ev):

    -8.4397  -7.0182  -5.3890  -3.0509  -0.1802   0.8666

     the Fermi energy is    -6.2088 ev

!    total energy              =      -4.25665619 Ry
     Harris-Foulkes estimate   =      -4.25665619 Ry
     estimated scf accuracy    <          9.2E-11 Ry

     total all-electron energy =        -4.256639 Ry

     The total energy is the sum of the following terms:

     one-electron contribution =      -2.55873208 Ry
     hartree contribution      =       1.54592834 Ry
     xc contribution           =      -2.26643381 Ry
     ewald contribution        =      -0.97584541 Ry
     one-center paw contrib.   =      -0.00154980 Ry
     smearing contrib. (-TS)   =      -0.00002343 Ry

     convergence has been achieved in  10 iterations

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.04930093    0.08539549    0.00000000
     atom    2 type  1   force =    -0.03215127   -0.05461178   -0.00000000
     atom    3 type  1   force =     0.03214644    0.05461530    0.00000000
     atom    4 type  1   force =    -0.04929610   -0.08539901   -0.00000000

     Total force =     0.165766     Total SCF correction =     0.000003


     Computing stress (Cartesian axis) and pressure

          total   stress  (Ry/bohr**3)                   (kbar)     P=  -31.37
  -0.00016778  -0.00026456   0.00000000        -24.68    -38.92      0.00
  -0.00026456  -0.00046181  -0.00000000        -38.92    -67.93     -0.00
   0.00000000  -0.00000000  -0.00001010          0.00     -0.00     -1.49


     init_run     :      1.29s CPU      1.41s WALL (       1 calls)
     electrons    :     12.07s CPU     12.79s WALL (       1 calls)
     forces       :      0.19s CPU      0.19s WALL (       1 calls)
     stress       :      2.47s CPU      2.53s WALL (       1 calls)'''

  examples = {
    'h4-nosym': {'in': h4nosym_in, 'out': h4nosym_out},
  }
  return examples[name]

def test_name_sep_val():
  eg = example_scf('h4-nosym')
  # parse input parameters
  ftmp = 'tmp.out'
  out_text = eg['in']
  with open('tmp.out', 'w') as f:
    f.write(out_text)
  mm = yq.read(ftmp)
  degauss = yq.name_sep_val(mm, 'degauss')
  assert np.isclose(degauss, 0.0063338)
  kstr = yq.name_sep_val(mm, 'K_POINTS', sep=None, dtype=str)
  assert kstr == 'automatic'
  mm.close()
  # parse output parameters
  ftmp = 'tmp.out'
  out_text = eg['out']
  with open('tmp.out', 'w') as f:
    f.write(out_text)
  mm = yq.read(ftmp)
  volume = yq.name_sep_val(mm, 'unit-cell volume')
  assert np.isclose(volume, 1330.2038)
  ram = yq.name_sep_val(mm, 'Estimated total dynamical RAM', '>')
  assert np.isclose(ram, 546.10)
  efermi = yq.name_sep_val(mm, 'the Fermi energy', 'is')
  assert np.isclose(efermi, -6.2088)
  niter = yq.name_sep_val(mm, 'convergence has been achieved', 'in', dtype=int)
  assert niter == 10
  ngvecs = yq.name_sep_val(mm, 'Dense  grid', ':')
  assert ngvecs == 45417
  mm.close()

def test_energy():
  eg = example_scf('h4-nosym')
  ftmp = 'tmp.out'
  out_text = eg['out']
  with open('tmp.out', 'w') as f:
    f.write(out_text)
  mm = yq.read(ftmp)
  e = yq.energy(mm)
  mm.close()
  assert np.isclose(e, -4.25665619)

def test_total_forces():
  eg = example_scf('h4-nosym')
  ftmp = 'tmp.out'
  out_text = eg['out']
  with open('tmp.out', 'w') as f:
    f.write(out_text)
  mm = yq.read(ftmp)
  forces = yq.total_forces(mm)
  mm.close()
  fref = np.array([
    [0.04930093, 0.08539549, 0],
    [-0.03215127, -0.05461178, 0],
    [0.03214644, 0.05461530, 0],
    [-0.04929610, -0.08539901, 0]
  ])
  assert np.allclose(forces, fref)

def test_total_stress():
  eg = example_scf('h4-nosym')
  ftmp = 'tmp.out'
  out_text = eg['out']
  with open('tmp.out', 'w') as f:
    f.write(out_text)
  mm = yq.read(ftmp)
  stress = yq.total_stress(mm)
  mm.close()
  sref = np.array([
    [-0.00016778, -0.00026456, 0],
    [-0.00026456, -0.00046181, 0],
    [0, 0, -0.00001010]
  ])
  assert np.allclose(stress, sref)

if __name__ == '__main__':
  test_name_sep_val()
  test_energy()
  test_total_forces()
  test_total_stress()
