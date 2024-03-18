
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
  [w]   # Diffusion potential (eV/mol)
    order = FIRST
    family = LAGRANGE
  []

 #### For Elasticity
  [disp_x]
    scaling=1.0e-05 
  []

  [disp_y]
    scaling=1.0e-05 
  []
[]

[ICs]
  [concentrationIC]   
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
  [right_x]
    type = DirichletBC
    variable = disp_x
    boundary = right
    value = 0
  []

  [left_x]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []     

  [top_x]
    type = DirichletBC
    variable = disp_x
    boundary = top
    value = 0
  []

  [bottom_x]
    type = DirichletBC
    variable = disp_x
    boundary = bottom
    value = 0
  []   

  [right_y]
    type = DirichletBC
    variable = disp_y
    boundary = right
    value = 0
  []

  [left_y]
    type = DirichletBC
    variable = disp_y
    boundary = left
    value = 0
  []     

  [top_y]
    type = DirichletBC
    variable = disp_y
    boundary = top
    value = 0
  []

  [bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []        
[]


[AuxVariables]
  [bnds]
  []
  [Energy]
    order = CONSTANT
    family = MONOMIAL
  []

  [von_mises]
    #Dependent variable used to visualize the Von Mises stress
    order = CONSTANT
    family = MONOMIAL
    # outputs = none
  []

  [stress_xx]
    order = CONSTANT
    family = MONOMIAL
  []

  [stress_yy]
    order = CONSTANT
    family = MONOMIAL
  []
[]

[AuxKernels]
  [von_mises_kernel]
    #Calculates the von mises stress and assigns it to von_mises
    type = RankTwoScalarAux
    variable = von_mises
    rank_two_tensor =stress
    execute_on = timestep_end
    scalar_type = VonMisesStress
  []

  [stress_xx]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xx
    index_i = 0
    index_j = 0
  []

  [stress_yy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yy
    index_i = 1
    index_j = 1
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

  [TensorMechanics]
    displacements = 'disp_x disp_y'
    # Plane Strain assumption means strain in the z-direction = 0
    planar_formulation = PLANE_STRAIN
    use_displaced_mesh = false
  []
[]

[Materials]
  # d is a scaling factor that makes it easier for the solution to converge
  # without changing the results. It is defined in each of the materials and
  # must have the same value in each one.
  [constants]
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
    f_name = Fch1
    # args = c
    coupled_variables = c
    constant_names =           'A      B      C        D     E      xe    Th         eV_J         d'
    # constant_expressions = '-9.6084 0.55496 24.701 -1.2624 1.7571 1.2029 10000  6.24150934e+18   1e-27'  ## For ~ -24,454 J/mol
    constant_expressions = ' -2.8245 0.60298 146.641 -2.1577 0.5918 0.3362 1e4    6.24150934e+18   1e-27'    ## For ~-24,885 J/mol
    function = 'eV_J*d*Th*(A*(E*c-xe)^2+B*(E*c-xe)+C*(E*c-xe)^6+D)'
    derivative_order = 2
  []

  [elasticity_tensor_B]   # 
    type = ComputeElasticityTensor
    base_name = matrix
    block=0
    fill_method = symmetric9
    # C_ijkl = '506 99 99 506 99 506 62 62 62'  
    # C_ijkl = '499 139 139 499 139 499 102 102 102'   # Cr -->GPa to ev/nm^3
    C_ijkl = '3114.51 867.57 867.57 3114.51 867.57 3114.51 636.63 636.63 636.63'
  []

  [const_stress_B]
    type = ComputeExtraStressConstant
    block = 0
    base_name = matrix
    extra_stress_tensor = '-0.288 -0.373 -0.2747 0 0 0'
  []

  [strain_B]  # 
    type = ComputeSmallStrain
    base_name = matrix
    block=0
    eigenstrain_names = eigenstrain
    displacements = 'disp_x disp_y'
  []

  [stress_B]
    type = ComputeLinearElasticStress
    base_name = matrix
    block=0
  []

  [eigenstrain_B]
    type = ComputeEigenstrain
    base_name = matrix
    block=0
    eigen_base = '0.1' #'3e-2' #'0.1 0.05 0 0 0 0.01'
    prefactor = -0 #pre # -1
    eigenstrain_name = eigenstrain
  []
  #### Mew Addition
  [pre_B]
    type = GenericConstantMaterial
    prop_names = pre1
    block=0
    #prop_values = 0.02
    prop_values = 0.002
  []

  [fel_etaB]      
    type = ElasticEnergyMaterial
    args = ' '
    base_name = matrix
    block=0
    f_name = fel1
    output_properties = fel1
    outputs = exodus
  []

  #################################################

  [elasticity_tensor_A]   # 
    type = ComputeElasticityTensor
    base_name = ppt
    block=0
    fill_method = symmetric9
    # C_ijkl = '289 154 154 289 154 289 80 80 80'  
    # C_ijkl = '196 83 52 196 52 251 56 56'  # Ti -->GPa to ev/nm^3
    C_ijkl = '1223.34 518.05 324.56 1223.34 324.56 1566.62 349.52 349.52 349.52'
  []
  [const_stress_A]
    type = ComputeExtraStressConstant
    block = 0
    base_name = ppt
    # extra_stress_tensor = '-0.288 -0.373 -0.2747 0 0 0'
    extra_stress_tensor = '-0.288 -0.5 -0.2747 0 0 0'
  []
  [strain_A]  # 
    type = ComputeSmallStrain
    base_name = ppt
    block=0
    eigenstrain_names = eigenstrain
    displacements = 'disp_x disp_y'
  []
  [stress_A]
    type = ComputeLinearElasticStress
    base_name = ppt
    block=0
  []
  [eigenstrain_A]
    type = ComputeEigenstrain
    base_name = ppt
    block=0
    eigen_base =  '0.1' #'3e-2' #'0.1 0.05 0 0 0 0.01'
    prefactor = -4 #pre # -1
    eigenstrain_name = eigenstrain
  []
  #### Mew Addition
  [pre_A]
    type = GenericConstantMaterial
    prop_names = pre2
    block=0
    prop_values = 0.02
    # prop_values = 0.002
  []

  [fel_etaA]      
    type = ElasticEnergyMaterial
    args = ' '
    base_name = ppt
    block=0
    f_name = fel2
    output_properties = fel2
    outputs = exodus
  []
  ##############################

  [global_stress]
    type = TwoPhaseStressMaterial
    base_A = ppt
    base_B = matrix
    block=0
  []


  [switching_function] #Interpolation function
    type = ParsedMaterial
    f_name = h
    args = 'c'
    # function =  '3*c1^2-2*c1^3'
    function =  '1/(1+exp(-10*(2*c-1)))'
  []

  # [switching]
  #   type = SwitchingFunctionMaterial
  #   eta = c1
  # []

  [F_1]
    type = DerivativeSumMaterial
    f_name = f_loc
    args = 'c'
    sum_materials = 'Fch1 fel1 fel2' #'Fch1 fel1'
    #sum_materials = 'fch0'
    outputs = exodus
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
  [von_mises]
    type = ElementAverageValue
    variable = von_mises
  []

    # Area of Phases
   [area_h]
       type = ElementIntegralMaterialProperty
       mat_prop = h
       execute_on = 'Initial TIMESTEP_END'
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
  petsc_options_iname = '-pc_type -ksp_gmres_restart -sub_ksp_type -sub_pc_type -pc_asm_overlap'
  petsc_options_value = 'asm 31 preonly ilu 1'
  #   dt = 500 #350
  dt = 500 #350
  # [TimeStepper]
  #   type = IterationAdaptiveDT
  #   dt = 500 #350
  #   cutback_factor = 0.5
  #   growth_factor = 1.5
  #   optimal_iterations = 7
  # []
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
  # file_base = Ti_Cr_elastic
  csv = true
  [my_checkpoint]
    type = Checkpoint
    num_files = 2
    interval = 2
    file_base = exodus_files/Ti_Cr_elastic
  []
  [console]
    type = Console
    max_rows = 10
  []
[]
