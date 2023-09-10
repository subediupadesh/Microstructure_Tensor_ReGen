
[Mesh]
  type = GeneratedMesh
  dim = 2
  elem_type = QUAD4
  nx = 100
  ny = 100
  nz = 0
  xmin = 0
  xmax = 50
  ymin = 0
  ymax = 50
  zmin = 0
  zmax = 0
  uniform_refine = 2
[]

[Variables]
  [c]   # Mole fraction of Au (unitless)
    order = FIRST
    family = LAGRANGE
  []
  [w]   # Chemical potential (eV/mol)
    order = FIRST
    family = LAGRANGE
  []
[]

[ICs]
  [concentrationIC]   # 50 mol% Au with variations
    type = RandomIC
    min = 0.7
    max = 0.9
    seed = 5
    variable = c
  []
[]

[BCs]
  [Periodic]
    [c_bcs]
      auto_direction = 'x y'
    []
  []
[]

[Kernels]
  [w_dot]
    variable = w
    v = c
    type = CoupledTimeDerivative
  []
  [coupled_res]
    variable = w
    type = SplitCHWRes
    mob_name = M
  []
  [coupled_parsed]
    variable = c
    type = SplitCHParsed
    f_name = f_loc
    kappa_name = kappa_c
    w = w
  []
[]

[Materials]
  # d is a scaling factor that makes it easier for the solution to converge
  # without changing the results. It is defined in each of the materials and
  # must have the same value in each one.
  [./constants]
    # Define constant values kappa_c and M. Eventually M will be replaced with
    # an equation rather than a constant.
    type = GenericFunctionMaterial
    prop_names = 'kappa_c M'
    prop_values = '6.297e-15*6.24150934e+18*1e+09^2*1e-27
                   1.981e-24*1e+09^2/6.24150934e+18/1e-27'
                   # kappa_c*eV_J*nm_m^2*d
                   # M*nm_m^2/eV_J/d  #1.981e-26
  []
  [local_energy]
    # Defines the function for the local free energy density as given in the
    # problem, then converts units and adds scaling factor.
    type = DerivativeParsedMaterial
    f_name = f_loc
    # args = c
    coupled_variables = c
    constant_names =           'A      B      C        D     E      xe    Th         eV_J         d'
    # constant_expressions = '-9.6084 0.55496 24.701 -1.2624 1.7571 1.2029 10000  6.24150934e+18   1e-27'  ## For ~ -24,454 J/mol
    constant_expressions = ' -2.8245 0.60298 146.641 -2.1577 0.5918 0.3362 1e4    6.24150934e+18   1e-27'    ## For ~-24,885 J/mol
    function = 'eV_J*d*Th*(A*(E*c-xe)^2+B*(E*c-xe)+C*(E*c-xe)^6+D)'
    derivative_order = 2
  []
[]

[Postprocessors]
  [step_size]             # Size of the time step
    type = TimestepSize
  []
  [iterations]            # Number of iterations needed to converge timestep
    type = NumNonlinearIterations
  []
  [nodes]                 # Number of nodes in mesh
    type = NumNodes
  []
  [evaluations]           # Cumulative residual calculations for simulation
    type = NumResidualEvaluations
  []
  [active_time]           # Time computer spent on simulation
    type = PerfGraphData
    section_name = "Root"
    data_type = total
  []
[]

[Preconditioning]
  [coupled]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  l_max_its = 30
  l_tol = 1e-6
  nl_max_its = 50
  nl_abs_tol = 1e-9
  end_time = 5000 #2592000   # 30 days
  petsc_options_iname = '-pc_type -ksp_gmres_restart -sub_ksp_type
                         -sub_pc_type -pc_asm_overlap'
  petsc_options_value = 'asm      31                  preonly
                         ilu          1'
  [TimeStepper]
    type = IterationAdaptiveDT
    dt = 500 #350
    cutback_factor = 0.8
    growth_factor = 5
    optimal_iterations = 7
  []
 # [Adaptivity]
 #   coarsen_fraction = 0.1
 #   refine_fraction = 0.7
 #   max_h_level = 2
 # []
[]

[Debug]
  show_var_residual_norms = true
[]

[Outputs]
  exodus = true
  console = true
  file_base = exodus_files/Ti_Cr
  # csv = true
  [console]
    type = Console
    max_rows = 10
  []
[]
