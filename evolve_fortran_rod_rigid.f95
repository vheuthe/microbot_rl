subroutine evolve_md_rod(mR, IR, X, Y, Theta, Xrod, Yrod, & 
                        distRod, act, Rm, Rr, dt, &
                        nsteps, tor, vel_act, vel_tor, &
                        ext_rod, cen_rod, mu,&
                        N, Nrod, &
                        new_XYT, new_XY_rod)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real,    intent(in) :: X(N), Y(N), Theta(N)
    real,    intent(in) :: Xrod(Nrod), Yrod(Nrod), distRod
    integer, intent(in) :: act(N)
    integer, intent(in) :: N, Nrod, nsteps
    real,    intent(in) :: Rm, Rr
    real,    intent(in) :: tor, vel_act, vel_tor, dt, mR, IR
    real,    intent(in) :: ext_rod, cen_rod , mu
    ! shape of rod is determined by factor of size of extremes and center
    ! =======================================
    real , intent(out) :: new_XYT(N,3), new_XY_rod(Nrod,2)
    ! =======================================
    real :: FX(N), FY(N), FR(N), v
    real :: F_pRX, F_pRY
    real :: F_Perp_X, F_Perp_Y, F_proj
    real :: mu_K_true = 0, mu_K =0, F_Perp, Friction
    real :: FXrod, FYrod
    real :: torquerod, rodXcm, rodYcm, rodtheta
    integer :: i, j, it
    real :: dx, dy, r2, drodx, drody 
    real, parameter :: PI = 3.14159265358979323846264
    ! =======================================
    ! force parameters
    real :: eps = 1., ss = 6.8, ss2, ss6, ss12, ff, epsRod=1.0
    ! =======================================
    ! rod parameters
    real :: Lrod2 = 0.d0, Lrod, fact(Nrod,4)
    ! =======================================
    ! viscous parameters
    real :: etaRot = 0.d0, etaLiq = 5.027, etaTra_per, etaTra_par, etaCol
    ! =======================================
    ss2  = ss*ss
    ss6  = ss2*ss2*ss2
    ss12 = ss6*ss6
    ! =======================================   

    new_XYT(:,1) = X
    new_XYT(:,2) = Y
    new_XYT(:,3) = Theta
    
    new_XY_rod(:,1) = Xrod
    new_XY_rod(:,2) = Yrod

    ! Nrod EVEN number!
    do i=1,Nrod/2
        fact(i,1) = ext_rod + (cen_rod-ext_rod)*abs((i-1)/(Nrod*1.))
        fact(i,2) = fact(i,1)*fact(i,1)
        fact(i,3) = fact(i,2)*fact(i,2)*fact(i,2)
        fact(i,4) = fact(i,3)*fact(i,3)
        fact(Nrod+1-i,1) = fact(i,1)
        fact(Nrod+1-i,2) = fact(i,2)
        fact(Nrod+1-i,3) = fact(i,3)
        fact(Nrod+1-i,4) = fact(i,4)
    enddo

    Lrod2 = (new_XY_rod(Nrod,2)-new_XY_rod(1,2))**2 + (new_XY_rod(Nrod,1)-new_XY_rod(1,1))**2
    Lrod = sqrt(Lrod2)
    
    rodXcm = SUM(X)/Nrod
    rodYcm = SUM(Y)/Nrod

    
    ! Diffusion for particles is determined by 
    ! Rm and Rr, which creates random forces.
    
    ! Diffusion for rod is determined by mR, which is the diffusion D0
    ! Diffusion is different in two directions: 
    
    ! mR is a scaling of friction. mR = 1 seems not to be working.

    etaCol     = etaLiq * 6 * PI * ss/2
    etaTra_par = etaLiq * 2 * PI * Lrod / log(Lrod/ss)
    etaTra_per = etaLiq * 4 * PI * Lrod / (log(Lrod/ss) + 1)
    etaRot     = etaLiq * PI * Lrod**3 / 3. /  log(Lrod/ss)

    ! if mu_K > 1 then there is friction alongside rod.
    mu_K = mu
    if (mu > 1) then
        mu_K_true = mu - 1 ! becomes real friction coefficient
        mu_K = 1
    endif
    

    do it = 1, nsteps

        FX = 0.d0
        FY = 0.d0
        FR = 0.d0
        
        FXrod = 0.d0
        FYrod = 0.d0
        torquerod = 0.d0
        
        rodtheta = atan2(new_XY_rod(Nrod,2)-new_XY_rod(1,2), new_XY_rod(Nrod,1)-new_XY_rod(1,1))
        ! =============================
        ! thermal motion
        ! =============================

        do i = 1, N
            FX(i) = gran()*Rm
            FY(i) = gran()*Rm
            FR(i) = gran()*Rr            
        enddo

        ! =============================
        ! repulsion between colloids
        ! =============================

        do i = 1, N-1
            do j = i+1, N
                dx = new_XYT(j,1) - new_XYT(i,1)
                dy = new_XYT(j,2) - new_XYT(i,2)
                r2 = dx*dx + dy*dy
                if (r2 < ss2) then
                    ff = ss12/(r2**6) - ss6/(r2**3)
                    ff = 12.*eps*ff/r2
                    ! ==== DEBUG ====
                    if (ff > 0.05 * etaCol / dt) then
                        print*, ff, sqrt(r2), 'colloids too much'
                        ff = 0.05 * etaCol / dt
                    endif
                    ! ===============
                    FX(i) = FX(i) - ff*dx
                    FY(i) = FY(i) - ff*dy
                    FX(j) = FX(j) + ff*dx
                    FY(j) = FY(j) + ff*dy                  
                endif 
            enddo
        enddo

        ! =============================
        ! repulsion with rod + activity
        ! =============================

        do i = 1, N

            ! ============================
            ! repulsion of single particle
            ! ============================
            
            F_Perp = 0.d0
            Friction = 0.d0
            
            do j = 1, Nrod
                drodx = new_XY_rod(j,1) - rodXcm
                drody = new_XY_rod(j,2) - rodYcm

                dx = new_XY_rod(j,1) - new_XYT(i,1)
                dy = new_XY_rod(j,2) - new_XYT(i,2)
         
                r2 = dx*dx + dy*dy
                  
                if (r2 < ss2*fact(j,2)) then
                    ff = fact(j,4)*ss12/(r2**6) - fact(j,3)*ss6/(r2**3)
                    ff = 12.*epsRod*fact(j,1)*ff/r2
                    ! ==== DEBUG ====
                    if (ff > 0.05 * etaCol / dt) then
                        print*, ff, sqrt(r2), 'rod too much'
                        ff = 0.05 * etaCol / dt
                    endif
                    ! ===============
                    ! F_particle = -(ff*dx, ff*dy)
                    !rel_theta = rodtheta - atan2(ff*dy, ff*dx)
                    
                    F_proj = (ff*dx)*cos(rodtheta) + (ff*dy)*sin(rodtheta)
                    
                    F_pRX = F_proj*cos(rodtheta)
                    F_pRY = F_proj*sin(rodtheta)
                    
                    F_Perp_X = (ff*dx - F_pRX)
                    F_Perp_Y = (ff*dy - F_pRY)
                    
                    F_Perp = F_Perp + sqrt(F_Perp_X**2 + F_Perp_Y**2)
                   
                    ! mu_K = 1       --> only perpendicular force are conserved.
                    ! mu_K = 0       --> some corrugation.
                    ! mu_K_true > 0  --> friction
                    
                    FX(i) = FX(i) - (ff*dx - F_pRX*mu_K)
                    FY(i) = FY(i) - (ff*dy - F_pRY*mu_K)

                    FXrod = FXrod + (ff*dx - F_pRY*mu_K)
                    FYrod = FYrod + (ff*dy - F_pRY*mu_K)       
                    
                    ! =======================
                    ! component of force in direction of rod
                    ! does not change torque.
                    ! =======================
                    
                    torquerod = torquerod + (ff*dy - F_pRY*mu_K)*drodx -&
                                            (ff*dx - F_pRX*mu_K)*drody

                endif 
            enddo

            ! ================================
            ! activity + friction
            ! ================================


            if (act(i)>0) then
                ! ========================
                ! Action includes rotation
                ! ========================
                v = vel_act
                if (act(i)>1) then
                    v = vel_tor
                    FR(i) = FR(i) - 2*tor*(act(i)-2.5)
                endif

                if (mu_K_true > 0) then
                    ! F_Perp * mu_K_true = maximum friction
                    ! Force parallel to rod is:
                    ! v*cos(new_XYT(i,3))*cos(rodtheta) + v*sin(new_XYT(i,3))*sin(rodtheta)
                    if (cos(rodtheta-new_XYT(i,3)) .ge. 0) then
                        Friction = + min(F_perp*mu_K_true, &
                            abs(etaCol*v*(cos(new_XYT(i,3))*cos(rodtheta) + sin(new_XYT(i,3))*sin(rodtheta)))) 
                    else 
                        Friction = - min(F_perp*mu_K_true, &
                            abs(etaCol*v*(cos(new_XYT(i,3))*cos(rodtheta) + sin(new_XYT(i,3))*sin(rodtheta)))) 
                    endif
                    
                endif


                FX(i) = FX(i) + cos(new_XYT(i,3))*v*etaCol - Friction*cos(rodtheta)
                FY(i) = FY(i) + sin(new_XYT(i,3))*v*etaCol - Friction*sin(rodtheta)

                FXrod = FXrod + Friction*cos(rodtheta)
                FYrod = FYrod + Friction*sin(rodtheta)

            endif
        enddo

        
        ! =============================
        ! Check how much was parallel force
        !if ((F_Perp_X .ne. 0).or.(F_Perp_Y .ne. 0)) then
        !    write(100,*), sqrt(F_Parallel_X**2 + F_Parallel_Y**2), sqrt(F_Perp_X**2 + F_Perp_Y**2),&
        !     atan(F_Parallel_Y/F_Parallel_X), atan(F_Perp_Y/F_Perp_X), rodtheta
        !endif

    
        ! =============================
        ! move
        ! =============================

        do i = 1, N
            new_XYT(i,1) = new_XYT(i,1) + FX(i)/etaCol*dt
            new_XYT(i,2) = new_XYT(i,2) + FY(i)/etaCol*dt
            new_XYT(i,3) = new_XYT(i,3) + FR(i)*dt
        enddo

        ! =============================
        ! move rod degrees of freedom
        ! =============================

        ! =============================
        ! Force is split in direction perpendicular and parallel to rod.
        ! Diffusivity is different for the two cases!
        ! =============================
        F_proj = FXrod*cos(rodtheta) + FYrod*sin(rodtheta)
        F_pRX = F_proj*cos(rodtheta)
        F_pRY = F_proj*sin(rodtheta)
        
        rodXcm = rodXcm + dt* (FXrod/etaTra_par + (FXrod - F_pRX)/etaTra_per)
        rodYcm = rodYcm + dt* (FYrod/etaTra_par + (FYrod - F_pRY)/etaTra_per)
        rodtheta = rodtheta + dt*torquerod/etaRot
        
        ! =============================
        ! transform rod 
        ! =============================

        do i = 1, Nrod
            new_XY_rod(i,1) = (i-(Nrod+1)/2.0)*cos(rodtheta)*distRod + rodXcm
            new_XY_rod(i,2) = (i-(Nrod+1)/2.0)*sin(rodtheta)*distRod + rodYcm
        enddo 
        
    enddo

contains 

    real FUNCTION gran()
    !     polar form of the Box-Muller transformation
    !     http://www.taygeta.com/random/gaussian.html
      implicit none
      !  real :: rand ! using old generator
      real :: rnum(2)
      real :: x1, x2, w, y1
      do
    !     x1 = 2.0 * rand() - 1.0 ! using old generator
    !     x2 = 2.0 * rand() - 1.0 ! using old generator
        call random_number(rnum)
        x1 = 2.0 * rnum(1) - 1.0
        x2 = 2.0 * rnum(2) - 1.0
        w = x1 * x1 + x2 * x2
        if( w < 1.0 ) goto 10
      end do
      10 continue
      w = sqrt( (-2.0 * log( w ) ) / w )
      y1 = x1 * w
      !  y2 = x2 * w ! there is here a second independent random number for free
      gran=y1
      return
    end function gran

end subroutine


subroutine  get_o_r_rod(X, Y, Theta, Xrod, Yrod, oldXrod, oldYrod, &
                        mode, rotDir, old_rotDir, &
                        flag_side, flag_LOS, &
                        ss, ssrod_ext, mR,&
                        ext_rod, cen_rod, &
                        obs_type, cones, cone_angle, &
                        Nobs, N, Nrod, Obs, Rew, touch) !DEBUG
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real, intent(in)    :: X(N), Y(N), Theta(N)
    real, intent(in)    :: Xrod(Nrod), Yrod(Nrod)
    real, intent(in)    :: oldXrod(Nrod), oldYrod(Nrod), cone_angle
    integer, intent(in) :: N, Nrod, Nobs, mode, rotDir, old_rotDir
    integer, intent(in) :: flag_side, obs_type, cones
    logical, intent(in) :: flag_LOS
    real, intent(out)   :: Obs(N, Nobs), Rew(N)
    integer :: i, j, k, n_cone, side
    integer :: iter_touch, adj(N,N) 
    integer, intent(out) :: touch(N)   
    real :: dx, dy, r, dtheta, val, th, cmRod(2), oldcmRod(2)
    real :: dx2, dy2, r2, dtheta2, dark, sp_th, ssrod, true_ss, true_ssrod
    real, intent(in) :: ss,  ssrod_ext, mR, ext_rod, cen_rod
    real :: covered_l, covered_r, vision_l, vision_r, in_sight=0., ss_touch=6.8
    real :: dRodtheta, dRod, rotRod, cone_angle_reduced, cone_slice, fact(Nrod)
    real, allocatable :: edge(:)
    real :: a, b, torque, near2(N), rod_L
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
  
    adj = 0
  
    cmRod(1) = SUM(Xrod)/Nrod
    cmRod(2) = SUM(Yrod)/Nrod

    true_ss = 6.0
    true_ssrod = sqrt((Xrod(1)-Xrod(2))**2 + (Yrod(1)-Yrod(2))**2)
    rod_L = true_ssrod * (Nrod - 1)
    
    ssrod = ssrod_ext    
    if (ssrod==0) ssrod = true_ssrod

    ! Nrod EVEN number!
    do i=1,Nrod/2
        fact(i) = ext_rod + (cen_rod-ext_rod)*abs((i-1)/(Nrod*1.))
        fact(Nrod+1-i) = fact(i)
    enddo

    oldcmRod(1) = SUM(oldXrod)/Nrod
    oldcmRod(2) = SUM(oldYrod)/Nrod

    dRod = sqrt((oldcmRod(2)-cmRod(2))**2 + (oldcmRod(1)-cmRod(1))**2 )


    dRodtheta = atan2(cmRod(2) - oldcmRod(2), cmRod(1) - oldcmRod(1))

    rotRod = atan2(   Yrod(Nrod)-   Yrod(1),   Xrod(Nrod)-   Xrod(1)) - &
             atan2(oldYrod(Nrod)-oldYrod(1),oldXrod(Nrod)-oldXrod(1))
    rotRod = rotRod / (2*PI) - floor(rotRod / (2*PI) + 0.5)

    ! cone_angle must be a positive angle in radiants
    allocate(edge(cones+1))
    do i = 0, cones
        edge(i+1) = -cone_angle/2. + cone_angle*i/cones  
    enddo 
    cone_slice = cone_angle / cones

    ! =============================
    ! CONSISTENCY CHECK ON N_OBS == 
    select case (mode)
        case (1)
            if (.not.(NObs == (2+flag_side)*cones)) then
                print*, 'ERROR consistency NObs'
                print*, 'NObs=', NObs, ' Should be =',(2+flag_side)*cones
                STOP
            endif
        case (2)
            if (.not.(NObs == (2+flag_side)*cones+2)) then
                print*, 'ERROR consistency NObs'
                print*, 'NObs=', NObs, ' Should be =',(2+flag_side)*cones+2
                STOP
            endif
        case (3) 
            if (.not.(NObs == (2+flag_side)*cones)) then
                print*, 'ERROR consistency  NObs'
                print*, 'NObs=', NObs, ' Should be =',(2+flag_side)*cones
                STOP
            endif
        case (4) 
            if (.not.(NObs == (2+flag_side)*cones+1)) then
                print*, 'ERROR consistency NObs'
                print*, 'NObs=', NObs, ' Should be =',(2+flag_side)*cones+1
                STOP
            endif
        case (5) 
            if (.not.(NObs == (2+flag_side)*cones)) then
                print*, 'ERROR consistency NObs'
                print*, 'NObs=', NObs, ' Should be =',(2+flag_side)*cones
                STOP
            endif
    end select
    ! =============================

    ! =============================
    ! seeing other particles ======
    ! =============================

    do i = 1, N-1
        do j = i+1, N
        
            !side = crossing(X(i),Y(i),X(j),Y(j),Xrod(1),&
            !                Yrod(1),Xrod(Nrod),Yrod(Nrod))
            
            side = 0
            ! side = 0 means on the same side of rod.  
            ! if flag_side == 1 then visibility is across rod.            
            if ((side == 0).or.(flag_side == 1)) then

                dx = X(j)-X(i)
                dy = Y(j)-Y(i)
                r = sqrt(dx*dx + dy*dy)
                ! check for adjajency and rewards
                if (r < ss_touch*1.25) then
                    adj(i,j) = 1
                    adj(j,i) = 1
                endif
                
                dtheta = atan2(dy,dx)
                sp_th = atan(ss, r)/2.
                ! i to j 
                ! th goes from [-pi, pi]
                th = (dtheta - Theta(i))/2./PI
                th = (th - floor(th + 0.5))*2*PI
                
                ! n_cone = 1 .. n_cone
                ! for theta in range [ -cone_angle , cone_angle]
                            
                if (obs_type == 1) then 
                val = (true_ss/r)
                else if (obs_type == 2) then
                    val = (true_ss/r**2)
                else 
                    print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                    STOP
                endif
                
                covered_l = 0
                covered_r = 0   
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                    ! terribly expensive way
                    ! to account for line of sight

                    if (flag_LOS) then
                        do k = 1, N 
                            
                            if ((i==k).or.(j==k)) cycle
                            
                            dx2 = X(k)-X(i)
                            dy2 = Y(k)-Y(i)
                            r2 = sqrt(dx2*dx2 + dy2*dy2)

                            if (r2 > r) cycle !only closer particles can obscure
                            
                            dtheta2 = atan2(dy2,dx2)
                            dtheta2 = (dtheta2 - Theta(i))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2 ! cone of shadow
                           

                            if (abs(th-dtheta2) < dark + sp_th) then
                                if (th .lt. dtheta2) then
                                    covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark)) 
                                else if (th .ge. dtheta2) then
                                    covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                                endif
                                
                                if (covered_l+covered_r > 2*sp_th) exit  ! fully covered
                            endif
                            
                        enddo
                    endif
                    
                    vision_l = th+sp_th-covered_l
                    vision_r = th-sp_th+covered_r

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                        
                        in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                        
                        Obs(i,n_cone+side*cones) = Obs(i,n_cone+side*cones)+val*in_sight
                    enddo
                    
                !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
                endif
                
                ! j to i
                th = (dtheta + PI - Theta(j))/2./PI
                ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
                th = (th - floor(th + 0.5))*2*PI
                covered_l = 0
                covered_r = 0   
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                    ! terribly expensive way
                    ! to account for line of sight
                    if (flag_LOS) then
                        do k = 1, N 
                            if ((i==k).or.(j==k)) cycle
                            dx2 = X(k)-X(j)
                            dy2 = Y(k)-Y(j)
                            r2 = sqrt(dx2*dx2 + dy2*dy2)
                            if (r2 > r) cycle
                            dtheta2 = atan2(dy2,dx2)
                            dtheta2 = (dtheta2 - Theta(j))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2.
                            
                            ! DTHETA AND DTHETA2 POSSIBLY NOT NORMALIZED
                            if (abs(th-dtheta2) < dark+sp_th) then
                                if (th .lt. dtheta2) then
                                    covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark))
                                else if (th .ge. dtheta2) then
                                    covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                                endif
                                
                                if (covered_l+covered_r > 2*sp_th) exit
                            endif
                            
                        enddo            
                    endif

                    vision_l = th+sp_th-covered_l
                    vision_r = th-sp_th+covered_r

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                                            
                        in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                        
                        Obs(j,n_cone+side*cones) = Obs(j,n_cone+side*cones)+val*in_sight
                                            
                    enddo
                endif
            endif
        enddo
    enddo

    ! check for rewards
    touch = 0
    near2 = 1000

    ! seeing the rod particles + rewards
    do i = 1, N

        a = Theta(i)  ! orientation of particle respect to x-axis.
        b = dRodtheta ! direction of motion of rod.
 
        do j = 1, Nrod
            dx = Xrod(j)-X(i)
            dy = Yrod(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            if (r < near2(i)) near2(i) = r
      
            dtheta = atan2(dy,dx)
            ! particle sees rod 
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI 
            sp_th = atan(ssrod, r)/2.
            ! -----------------------------
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
            ! print*, X(i), Y(i), Theta(i), Xrod(j), Yrod(j), th, n_cone

            if (obs_type == 1) then 
                val = (true_ssrod)/r*fact(i)
            else if (obs_type == 2) then
                val = (true_ssrod/r**2)*fact(i)
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
                        
            covered_l = 0
            covered_r = 0   
            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                ! terribly expensive way
                ! to account for line of sight

                if (flag_LOS) then
                    do k = 1, N 
                        
                        dx2 = X(k)-X(i)
                        dy2 = Y(k)-Y(i)
                        r2 = sqrt(dx2*dx2 + dy2*dy2)

                        if (r2 > r) cycle !only closer particles can obscure
                        
                        dtheta2 = atan2(dy2,dx2)
                        dtheta2 = (dtheta2 - Theta(i))/2./PI
                        dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                        dark = atan(ss, r2)/2 ! cone of shadow
                       

                        if (abs(th-dtheta2) < dark + sp_th) then
                            if (th .lt. dtheta2) then
                                covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark)) 
                            else if (th .ge. dtheta2) then
                                covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                            endif
                            
                            if (covered_l+covered_r > 2*sp_th) exit  ! fully covered
                        endif
                        
                    enddo
                endif
                
                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                    Obs(i,n_cone+(1+flag_side)*cones) = Obs(i,n_cone+(1+flag_side)*cones)+val*in_sight
                enddo


            endif
            if (near2(i) <= ss_touch*1.25) touch(i) = 1
        enddo
        
    enddo
    
    iter_touch = 1
    do while (iter_touch == 1)
        iter_touch = 0
        do i = 1, N
            if (touch(i) == 1) cycle
            do j = 1, N
                if ( ( adj(i,j) == 1) .and. (touch(j)==1)) then
                    iter_touch = 1
                    touch(i) = 1
                    exit
                endif
            enddo
        enddo
    enddo
        
    do i = 1, N
        a = Theta(i)
        b = dRodtheta
        dx = cmRod(1) - X(i)
        dy = cmRod(2) - Y(i)
        r = sqrt(dx*dx + dy*dy)
        torque = cos(a)*dy - sin(a)*dx

        
        ! different reward functions to choose from
        select case (mode)
            case (1)
                Rew(i) = reward_move(r/true_ss, dRod, a, b, rotRod, touch(i))
                !print*, i, 'x ', X(i),'y ', Y(i), ' theta ', a, 'rodtheta ', b,&
                !       'a-b ', a-b, ' mod2pi ', ((a-b) - floor((a-b)/2.d0/PI+0.5d0)*2*PI), &
                !       reward_move(r/ss, dRod, a, b, near)
            case (2)
                Rew(i) = reward_move_back(r/true_ss, dRod, a, b, touch(i))
                Obs(i, (2+flag_side)*cones+1) = cos(a)
                Obs(i, (2+flag_side)+2) = sin(a)
            case (3) 
                ! reward positive irrespective to direction of rotation
                if (sum(Obs(i, ((1+flag_side)*cones+(cones+1)/2):&
                               ((1+flag_side)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(rotRod, torque, touch(i), 0.0) 
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                endif
                
            case (4) 
                ! reward positive only if clockwise (-1) or anti-clockwise (+1).
                ! no penalty for translation of center of mass.
                ! NON - NEGATIVE REWARD
                                ! reward positive irrespective to direction of rotation
                if (sum(Obs(i, ((1+flag_side)*cones+(cones+1)/2):&
                               ((1+flag_side)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(abs(rotRod), torque*old_rotDir, touch(i), dRod * 24. / rod_L) * mR 
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                    Obs(i, (2+flag_side)+1) = rotDir
                endif
            case (5) ! debug reward for contact
                Rew(i) = r/true_ss * touch(i)
        end select
        
        Rew(i) = Rew(i) - (tanh((near2(i)-10*ss_touch)/10)+1)/5
    enddo

    return

contains

    real FUNCTION ff(x,y,x1,y1,x2,y2)
      implicit none 
      real :: x1,y1,x2,y2,x,y
      ff = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
      return 
    end function

    integer FUNCTION crossing(x1,y1,x2,y2,rx1,ry1,rx2,ry2)
    ! returns value 1 if particles are on the other side of rod
      implicit none
      real :: x1,y1,x2,y2,rx1,ry1,rx2,ry2
      crossing = 0
      if (ff(rx1,ry1,x1,y1,x2,y2)*ff(rx2,ry2,x1,y1,x2,y2)< 0 ) then
        if (ff(x1,y1,rx1,ry1,rx2,ry2)*ff(x2,y2,rx1,ry1,rx2,ry2)< 0 ) then
          crossing = 1
        endif
      endif
    end function

    real FUNCTION reward_rotate(rotRod, tq, near, dRod)
    ! reward function for rotation on the spot
      implicit none
      !
      real :: rotRod, tq, dRod
      integer :: near

      reward_rotate =  ((rotRod * tq) - dRod) * near
      return
    end function reward_rotate    

    real FUNCTION reward_move_back(rss, dRod, a, b, near)
    ! reward function for linear translation in direction (-x)
      implicit none 
      !
      real :: rss, a, b, dRod
      integer :: near
      reward_move_back =  -cos(b) * dRod * cos(a)**2 / rss * near * 10.
      return
    end function reward_move_back    
    
    real FUNCTION reward_move(rss, dRod, a, b, rot, near)
    ! Reward function for linear translation in any direction. 
    ! Maximum reward when particle is aligned with rod direction.
    ! substracts a cost for rotation
      implicit none
      real :: rss, a, b, dRod, ab_half, rot
      integer :: near
      ab_half = ((a-b) - floor((a-b)/2./PI+0.5)*2*PI)/2.
      reward_move = (dRod * cos(ab_half) / rss * near  - abs(rot)) * 10.
      return
    end function reward_move
    
end subroutine


! ===============================================================

subroutine  get_o_r_rod_differential(X, Y, Theta, Xrod, Yrod, oldXrod, oldYrod, &
                        mode, rotDir, old_rotDir, &
                        flag_diff, flag_LOS, &
                        ss, ssrod_ext, mR,&
                        obs_type, cones, cone_angle, &
                        Nobs, N, Nrod, Obs, Rew, touch) !DEBUG
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real, intent(in)    :: X(N), Y(N), Theta(N)
    real, intent(in)    :: Xrod(Nrod), Yrod(Nrod)
    real, intent(in)    :: oldXrod(Nrod), oldYrod(Nrod), cone_angle
    integer, intent(in) :: N, Nrod, Nobs, mode, rotDir, old_rotDir
    integer, intent(in) :: flag_diff, obs_type, cones
    logical, intent(in) :: flag_LOS
    real, intent(out)   :: Obs(N, Nobs), Rew(N)
    integer :: i, j, k, n_cone, side
    integer :: iter_touch, adj(N,N) 
    integer, intent(out) :: touch(N)   
    real :: dx, dy, r, dtheta, val, th, cmRod(2), oldcmRod(2)
    real :: dx2, dy2, r2, dtheta2, dark, sp_th, ssrod, true_ss, true_ssrod
    real, intent(in) :: ss,  ssrod_ext, mR
    real :: covered_l, covered_r, vision_l, vision_r, in_sight=0., ss_touch=6.8
    real :: dRodtheta, dRod, rotRod, cone_angle_reduced, cone_slice
    real, allocatable :: edge(:)
    real :: a, b, torque, near2(N), rod_L
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
  
    adj = 0
  
    cmRod(1) = SUM(Xrod)/Nrod
    cmRod(2) = SUM(Yrod)/Nrod

    true_ss = 6.0
    true_ssrod = sqrt((Xrod(1)-Xrod(2))**2 + (Yrod(1)-Yrod(2))**2)
    rod_L = true_ssrod * (Nrod - 1)
    
    ssrod = ssrod_ext    
    if (ssrod==0) ssrod = true_ssrod

    oldcmRod(1) = SUM(oldXrod)/Nrod
    oldcmRod(2) = SUM(oldYrod)/Nrod

    dRod = sqrt((oldcmRod(2)-cmRod(2))**2 + (oldcmRod(1)-cmRod(1))**2 )


    dRodtheta = atan2(cmRod(2) - oldcmRod(2), cmRod(1) - oldcmRod(1))

    rotRod = atan2(   Yrod(Nrod)-   Yrod(1),   Xrod(Nrod)-   Xrod(1)) - &
             atan2(oldYrod(Nrod)-oldYrod(1),oldXrod(Nrod)-oldXrod(1))
    rotRod = rotRod / (2*PI) - floor(rotRod / (2*PI) + 0.5)

    ! cone_angle must be a positive angle in radiants
    allocate(edge(cones+1))
    do i = 0, cones
        edge(i+1) = -cone_angle/2. + cone_angle*i/cones  
    enddo 
    cone_slice = cone_angle / cones

    ! =============================
    ! CONSISTENCY CHECK ON N_OBS == 
    select case (mode)
        case (1)
            if (.not.(NObs == (2+flag_diff)*cones)) then
                print*, 'ERROR consistency NObs'
                STOP
            endif
        case (2)
            if (.not.(NObs == (2+flag_diff)*cones+2)) then
                print*, 'ERROR consistency NObs'
                STOP
            endif
        case (3) 
            if (.not.(NObs == (2+flag_diff)*cones)) then
                print*, 'ERROR consistency  NObs'
                STOP
            endif
        case (4) 
            if (.not.(NObs == (2+flag_diff)*cones+1)) then
                print*, 'ERROR consistency NObs'
                STOP
            endif
        case (5) 
            if (.not.(NObs == (2+flag_diff)*cones)) then
                print*, 'ERROR consistency NObs'
                STOP
            endif
    end select
    ! =============================

    ! =============================
    ! seeing other particles ======
    ! =============================

    do i = 1, N-1
        do j = i+1, N
        
            !side = crossing(X(i),Y(i),X(j),Y(j),Xrod(1),&
            !                Yrod(1),Xrod(Nrod),Yrod(Nrod))
            
            side = 0
            ! side = 0 means on the same side of rod.  
            if ((side == 0)) then

                dx = X(j)-X(i)
                dy = Y(j)-Y(i)
                r = sqrt(dx*dx + dy*dy)
                ! check for adjajency and rewards
                if (r < ss_touch*1.25) then
                    adj(i,j) = 1
                    adj(j,i) = 1
                endif
                
                dtheta = atan2(dy,dx)
                sp_th = atan(ss, r)/2.
                ! i to j 
                ! th goes from [-pi, pi]
                th = (dtheta - Theta(i))/2./PI
                th = (th - floor(th + 0.5))*2*PI
                
                ! n_cone = 1 .. n_cone
                ! for theta in range [ -cone_angle , cone_angle]
                            
                if (obs_type == 1) then 
                val = (true_ss/r)
                else if (obs_type == 2) then
                    val = (true_ss/r**2)
                else 
                    print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                    STOP
                endif
                
                covered_l = 0
                covered_r = 0   
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                    ! terribly expensive way
                    ! to account for line of sight

                    if (flag_LOS) then
                        do k = 1, N 
                            
                            if ((i==k).or.(j==k)) cycle
                            
                            dx2 = X(k)-X(i)
                            dy2 = Y(k)-Y(i)
                            r2 = sqrt(dx2*dx2 + dy2*dy2)

                            if (r2 > r) cycle !only closer particles can obscure
                            
                            dtheta2 = atan2(dy2,dx2)
                            dtheta2 = (dtheta2 - Theta(i))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2 ! cone of shadow
                           

                            if (abs(th-dtheta2) < dark + sp_th) then
                                if (th .lt. dtheta2) then
                                    covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark)) 
                                else if (th .ge. dtheta2) then
                                    covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                                endif
                                
                                if (covered_l+covered_r > 2*sp_th) exit  ! fully covered
                            endif
                            
                        enddo
                    endif
                    
                    vision_l = th+sp_th-covered_l
                    vision_r = th-sp_th+covered_r

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                        
                        in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                        
                        Obs(i,n_cone+side*cones) = Obs(i,n_cone+side*cones)+val*in_sight
                    enddo
                    
                !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
                endif
                
                ! j to i
                th = (dtheta + PI - Theta(j))/2./PI
                ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
                th = (th - floor(th + 0.5))*2*PI
                covered_l = 0
                covered_r = 0   
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                    ! terribly expensive way
                    ! to account for line of sight
                    if (flag_LOS) then
                        do k = 1, N 
                            if ((i==k).or.(j==k)) cycle
                            dx2 = X(k)-X(j)
                            dy2 = Y(k)-Y(j)
                            r2 = sqrt(dx2*dx2 + dy2*dy2)
                            if (r2 > r) cycle
                            dtheta2 = atan2(dy2,dx2)
                            dtheta2 = (dtheta2 - Theta(j))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2.
                            
                            ! DTHETA AND DTHETA2 POSSIBLY NOT NORMALIZED
                            if (abs(th-dtheta2) < dark+sp_th) then
                                if (th .lt. dtheta2) then
                                    covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark))
                                else if (th .ge. dtheta2) then
                                    covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                                endif
                                
                                if (covered_l+covered_r > 2*sp_th) exit
                            endif
                            
                        enddo            
                    endif

                    vision_l = th+sp_th-covered_l
                    vision_r = th-sp_th+covered_r

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                                            
                        in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                        
                        Obs(j,n_cone+side*cones) = Obs(j,n_cone+side*cones)+val*in_sight
                                            
                    enddo
                endif
            endif
        enddo
    enddo

    ! check for rewards
    touch = 0
    near2 = 1000

    ! seeing the rod particles + rewards
    do i = 1, N

        a = Theta(i)  ! orientation of particle respect to x-axis.
        b = dRodtheta ! direction of motion of rod.
 
        do j = 1, Nrod
            dx = Xrod(j)-X(i)
            dy = Yrod(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            if (r < near2(i)) near2(i) = r
      
            dtheta = atan2(dy,dx)
            ! particle sees rod 
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI 
            sp_th = atan(ssrod, r)/2.
            ! -----------------------------
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
            ! print*, X(i), Y(i), Theta(i), Xrod(j), Yrod(j), th, n_cone

            if (obs_type == 1) then 
                val = (true_ssrod)/r
            else if (obs_type == 2) then
                val = (true_ssrod/r**2)
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
                        
            covered_l = 0
            covered_r = 0   
            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                ! terribly expensive way
                ! to account for line of sight

                if (flag_LOS) then
                    do k = 1, N 
                        
                        dx2 = X(k)-X(i)
                        dy2 = Y(k)-Y(i)
                        r2 = sqrt(dx2*dx2 + dy2*dy2)

                        if (r2 > r) cycle !only closer particles can obscure
                        
                        dtheta2 = atan2(dy2,dx2)
                        dtheta2 = (dtheta2 - Theta(i))/2./PI
                        dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                        dark = atan(ss, r2)/2 ! cone of shadow
                       

                        if (abs(th-dtheta2) < dark + sp_th) then
                            if (th .lt. dtheta2) then
                                covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark)) 
                            else if (th .ge. dtheta2) then
                                covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                            endif
                            
                            if (covered_l+covered_r > 2*sp_th) exit  ! fully covered
                        endif
                        
                    enddo
                endif
                
                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                    ! orientational divided observations
                    Obs(i,2*n_cone-1) = Obs(i,2*n_cone-1) + val*in_sight*cos(th)
                    Obs(i,2*n_cone)   = Obs(i,2*n_cone)   + val*in_sight*sin(th)
                enddo


            endif
            if (near2(i) <= ss_touch*1.25) touch(i) = 1
        enddo
        
    enddo
    
    iter_touch = 1
    do while (iter_touch == 1)
        iter_touch = 0
        do i = 1, N
            if (touch(i) == 1) cycle
            do j = 1, N
                if ( ( adj(i,j) == 1) .and. (touch(j)==1)) then
                    iter_touch = 1
                    touch(i) = 1
                    exit
                endif
            enddo
        enddo
    enddo
        
    do i = 1, N
        a = Theta(i)
        b = dRodtheta
        dx = cmRod(1) - X(i)
        dy = cmRod(2) - Y(i)
        r = sqrt(dx*dx + dy*dy)
        torque = cos(a)*dy - sin(a)*dx

        
        ! different reward functions to choose from
        select case (mode)
            case (1)
                Rew(i) = reward_move(r/true_ss, dRod, a, b, rotRod, touch(i))
                !print*, i, 'x ', X(i),'y ', Y(i), ' theta ', a, 'rodtheta ', b,&
                !       'a-b ', a-b, ' mod2pi ', ((a-b) - floor((a-b)/2.d0/PI+0.5d0)*2*PI), &
                !       reward_move(r/ss, dRod, a, b, near)
            case (2)
                Rew(i) = reward_move_back(r/true_ss, dRod, a, b, touch(i))
                Obs(i, (2+flag_diff)*cones+1) = cos(a)
                Obs(i, (2+flag_diff)+2) = sin(a)
            case (3) 
                ! reward positive irrespective to direction of rotation
                ! no penalty for translation of center of mass.
                if (sum(Obs(i, ((1+flag_diff)*cones+(cones+1)/2):&
                               ((1+flag_diff)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(rotRod, torque, touch(i), dRod * 24. / rod_L) * mR 
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                endif
                
            case (4) 
                ! reward positive only if clockwise (-1) or anti-clockwise (+1).
                ! no penalty for translation of center of mass.
                ! NON - NEGATIVE REWARD
                                ! reward positive irrespective to direction of rotation
                if (sum(Obs(i, ((1+flag_diff)*cones+(cones+1)/2):&
                               ((1+flag_diff)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(abs(rotRod), torque*old_rotDir, touch(i), dRod * 24. / rod_L) * mR 
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                    Obs(i, (2+flag_diff)+1) = rotDir
                endif
                
            case (5) ! debug reward for contact
                Rew(i) = r/true_ss * touch(i)
        end select
        
        Rew(i) = Rew(i) - (tanh((near2(i)-10*ss_touch)/10)+1)/5
    enddo

    return

contains

    real FUNCTION ff(x,y,x1,y1,x2,y2)
      implicit none 
      real :: x1,y1,x2,y2,x,y
      ff = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
      return 
    end function

    integer FUNCTION crossing(x1,y1,x2,y2,rx1,ry1,rx2,ry2)
    ! returns value 1 if particles are on the other side of rod
      implicit none
      real :: x1,y1,x2,y2,rx1,ry1,rx2,ry2
      crossing = 0
      if (ff(rx1,ry1,x1,y1,x2,y2)*ff(rx2,ry2,x1,y1,x2,y2)< 0 ) then
        if (ff(x1,y1,rx1,ry1,rx2,ry2)*ff(x2,y2,rx1,ry1,rx2,ry2)< 0 ) then
          crossing = 1
        endif
      endif
    end function

    real FUNCTION reward_rotate(rotRod, tq, near, dRod)
    ! reward function for rotation on the spot
      implicit none
      !
      real :: rotRod, tq, dRod
      integer :: near

      reward_rotate =  ((rotRod * tq) - dRod) * near
      return
    end function reward_rotate    

    real FUNCTION reward_move_back(rss, dRod, a, b, near)
    ! reward function for linear translation in direction (-x)
      implicit none 
      !
      real :: rss, a, b, dRod
      integer :: near
      reward_move_back =  -cos(b) * dRod * cos(a)**2 / rss * near * 10.
      return
    end function reward_move_back    
    
    real FUNCTION reward_move(rss, dRod, a, b, rot, near)
    ! Reward function for linear translation in any direction. 
    ! Maximum reward when particle is aligned with rod direction.
    ! substracts a cost for rotation
      implicit none
      real :: rss, a, b, dRod, ab_half, rot
      integer :: near
      ab_half = ((a-b) - floor((a-b)/2./PI+0.5)*2*PI)/2.
      reward_move = (dRod * cos(ab_half) / rss * near  - abs(rot)) * 10.
      return
    end function reward_move
    
end subroutine