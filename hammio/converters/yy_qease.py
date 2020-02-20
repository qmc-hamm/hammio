# Author: Yubo "Paul" Yang
# Email: yubo.paul.yang@gmail.com
# Convert QE output to ASE objects, hope to eventually merge to ase.io
from mmap import mmap
import numpy as np

# ==================== level 0: general routines =====================
def read(scf_out):
  """Get a memory map pointer to file

  Args:
    fname (str): filename
  Return:
    mmap: memory map to file
  Example:
    >>> mm = read('scf.out')
  """
  with open(scf_out, 'r+') as f:
    mm = mmap(f.fileno(), 0)
  return mm

def stay(read_func, *args, **kwargs):
  """Stay at current memory location after read

  Args:
    Callable: read function, which takes mmap as first input
  Return:
    Callable: read but no change to mmap tell()
  """
  def wrapper(mm, *args, **kwargs):
    idx = mm.tell()
    ret = read_func(mm, *args, **kwargs)
    mm.seek(idx)
    return ret
  return wrapper

@stay
def name_sep_val(mm, name, sep='=', dtype=float, ipos=1):
  """Read key-value pair such as "name = value"

  Args:
    mm (mmap): memory map
    name (str): name of variable; used to find value line
    sep (str, optional): separator, default '='
    dtype (type, optional): variable data type, default float
    ipos (int, optiona): position of value in line, default 1 (after sep)
  Return:
    dtype: value of requested variable
  Examples:
    >>> name_sep_val(mm, 'a')  # 'a = 2.4'
    >>> name_sep_val(mm, 'volume', ipos=-2) # 'volume = 100.0 bohr^3'
    >>> name_sep_val(mm, 'key', sep=':')  # 'key:val'
    >>> name_sep_val(mm, 'new', sep=':')  # 'new:name'
    >>> name_sep_val(mm, 'natom', dtype=int)  # 'new:name'
  """
  idx = mm.find(name.encode())
  if idx == -1:
    raise RuntimeError('"%s" not found' % name)
  mm.seek(idx)
  line = mm.readline()
  tokens = line.split(sep.encode())
  val_text = tokens[ipos].split()[0]
  val = dtype(val_text)
  return val

# ======================== level 1: parse QE =========================
units = dict(  # CODATA 2018; convert to Angstrom, eV
  ry = 27.211386245988/2,
  bohr = 0.529177210903,
)

@stay
def energy(mm):
  """Read total energy

  Args:
    mm (mmap): memory map
  Return:
    float: first total energy in Ry
  Example:
    >>> mm = read('scf.out')
    >>> e = energy(mm)
  """
  idx = mm.find(b'!')
  mm.seek(idx)
  eline = mm.readline()
  energy = float(eline.split()[-2])
  return energy

@stay
def total_forces(mm, ndim=3):
  """Read total forces

  Args:
    mm (mmap): memory map to file
    ndim (int, optional): number of spatial dimensions, default 3
  Return:
    np.array: forces in Ry/Bohr, shape (natom, ndim)
  Example:
    >>> mm = read('scf.out')
    >>> forces = total_forces(mm)
  """
  # allocate memory
  natom = name_sep_val(mm, 'number of atoms', dtype=int)
  forces = np.zeros([natom, ndim])
  # find forces text block
  begin_tag = 'Forces acting on atoms'
  end_tag = 'The non-local contrib.  to forces'
  begin_idx = mm.find(begin_tag.encode())
  end_idx = mm.find(end_tag.encode())
  force_block = mm[begin_idx:end_idx].decode()
  # parse forces text block
  iatom = 0
  for line in force_block.split('\n'):
    if line.strip().startswith('atom'):
      tokens = line.split()
      if len(tokens) == 6+ndim:  # found an atom
        forces[iatom, :] = list(map(float, tokens[-ndim:]))
        iatom += 1
  return forces

@stay
def total_stress(mm, ndim=3):
  """Read total stress

  Args:
    mm (mmap): memory map to file
    ndim (int, optional): number of spatial dimensions, default 3
  Return:
    np.array: stress in Ry/Bohr^3, shape (ndim, ndim)
  Example:
    >>> mm = read('scf.out')
    >>> stress = total_stress(mm)
  """
  # allocate memory
  stress = np.zeros([ndim, ndim])
  # find stress start
  begin_tag = 'total   stress  (Ry/bohr**3)'
  idx = mm.find(begin_tag.encode())
  mm.seek(idx)
  mm.readline()  # skip title
  # parse stress
  for idim in range(ndim):
    line = mm.readline()
    tokens = line.split()
    if not len(tokens) == 2*ndim:
      msg = 'cannot parse stress "%s"\n' % line
      msg += ' is ndim=%d correct?' % ndim
      raise RuntimeError(msg)
    stress[idim, :] = list(map(float, tokens[:ndim]))
  return stress

# ====================== level 2: construct ASE =======================

def calculator_from_results(atoms, results):
  from ase.calculators.singlepoint import SinglePointCalculator
  # add "virial" to calculators
  from ase.calculators import calculator
  calculator.all_properties.append('virial')
  # check that all results can be saved
  msg = ''
  for key in results.keys():
    if key not in calculator.all_properties:
      msg += '%s not supported\n' % key
  if msg != '':
    raise RuntimeError(msg)
  # make calculator
  calc = SinglePointCalculator(atoms, **results)
  return calc

def qe2ase(scf_in, scf_out):
  from ase import io
  atoms = io.read(scf_in)
  # read results
  mm = read(scf_out)
  ene = energy(mm)*units['ry']
  forces = total_forces(mm)*units['ry']/units['bohr']
  stress = total_stress(mm)*units['ry']/units['bohr']**3
  virial = stress*atoms.get_volume()
  results = {
    'energy': ene,
    'forces': forces,
    'virial': virial
  }
  # complete ASE object
  calc = calculator_from_results(atoms, results)
  atoms.set_calculator(calc)
  return atoms
