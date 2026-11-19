C    This file has been generated using SystemGenerator version 8.0.2                                                                           
C  Parameters for unit and force center list                                                                                                    
       integer maxnch          !Maximum number of molecules + atoms                                                                             
          parameter (maxnch= 768)                                                                                                               
       integer maxat           !Maximum number of force centers                                                                                 
          parameter (maxat= 17544)                                                                                                              
C                                                                                                                                               
C  Parameters for connectivity lists                                                                                                            
       integer maxbonds        !Maximum number of bonds in system                                                                               
          parameter (maxbonds= 18656)                                                                                                           
       integer maxbends        !Maximum number of bends in system                                                                               
          parameter (maxbends= 30992)                                                                                                           
       integer maxtorts        !Maximum number of torsions in system                                                                            
          parameter (maxtorts= 41424)                                                                                                           
       integer maxdeforms      !Maximum number of deformations in system                                                                        
         parameter (maxdeforms= 10752)                                                                                                          
C                                                                                                                                               
C  Parameters for force field                                                                                                                   
       integer maxtypes        !Max number of repulsion-dispersion types                                                                        
          parameter (maxtypes= 12)                                                                                                              
       integer maxcharges      !Max number of charge types                                                                                      
          parameter (maxcharges= 27)                                                                                                            
       integer maxdummy        !Max number of dummy (lone pair) atoms                                                                           
          parameter (maxdummy= 2000)                                                                                                            
       integer maxbtypes       !Max of types (bonds,bends,torts)                                                                                
          parameter (maxbtypes= 33)                                                                                                             
       integer maxfolds        !Max number of terms in torsional energy                                                                         
          parameter (maxfolds = 101 )                                                                                                           
       integer maxbox                                                                                                                           
       parameter (maxbox = 200)                                                                                                                 
       integer maxnay          !Maximum neighbors per force center                                                                              
          parameter (maxnay = 2000)                                                                                                             
       integer maxdim          !Maximum box subdivisions per axis                                                                               
          parameter (maxdim = 100)                                                                                                              
       integer maxdim3         !Total number of box subdivisions                                                                                
          parameter (maxdim3 = maxdim*maxdim*maxdim)                                                                                            
C                                                                                                                                               
C  Polarization related parameters                                                                                                              
       real*8 a_thole          !Dipole-Dipole damping constant                                                                                  
          parameter (a_thole=0.20d0)                                                                                                            
C                                                                                                                                               
C  Misc additional parameters as indicated (advanced)                                                                                           
       integer maxpoints       !Max points in interaction splines                                                                               
          parameter (maxpoints = 2000)                                                                                                          
       integer maxprop         !Max number of system averaged properties                                                                        
          parameter (maxprop = 40)                                                                                                              
       integer maxnnb          !Max number of non-bonded interactions                                                                           
          parameter (maxnnb = maxcharges*(maxcharges+1)/2)                                                                                      
C                                                                                                                                               
C  Parameters for all parallel code:                                                                                                            
       integer kmax_x, kmax_y, kmax_z                                                                                                           
         parameter (kmax_x=7)                                                                                                                   
         parameter (kmax_y=7)                                                                                                                   
         parameter (kmax_z=16)                                                                                                                   
C                                                                                                                                               
       integer iChainSurf                                                                                                                       
         parameter(iChainSurf=maxnch+1) ! the last Surf chain to unconstrain bonds                                                              
       integer Nzslice                  ! number of slices for the surface                                                                      
         parameter (Nzslice=120*30)                                                                                                             
       real*8 P2up                      ! How often to update the density profile in fs                                                         
         parameter (P2up=50000.0d0)                                                                                                             
C                                                                                                                                               
C  Parameters for atomic decomposition parallel code.                                                                                           
      integer ncpu                                                                                                                              
         parameter (ncpu =    40)
       integer maxat_tot, maxat_pr
         parameter (maxat_tot=maxat)
         parameter (maxat_pr =       826)
