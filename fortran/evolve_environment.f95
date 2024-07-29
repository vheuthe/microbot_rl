subroutine evolve_md_rod(mR, IR, X, Y, Theta, &
                        oldNoise, old_vel_noise, old_tor_noise, &
                        Xrod, Yrod, Xobst, Yobst, &
                        distRod, act, Rm, Rr, dt, &
                        tor, vel_act, vel_tor, vel_noise_fact, rot_noise_fact, &
                        ext_rod, cen_rod, mu, reproduction, &
                        noiseFlag, obst_flag, N, Nrod, Nobst, nsteps, &
                        new_XYT, new_XY_rod, part_rod_forces, &
                        noise, vel_noise, tor_noise)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real,    intent(in) :: X(N), Y(N), Theta(N)
    real,    intent(in) :: oldNoise(N, 3 * nsteps), old_vel_noise(N, nsteps), old_tor_noise(N, nsteps)
    real,    intent(in) :: Xrod(Nrod), Yrod(Nrod), distRod
    real,    intent(in) :: Xobst(Nobst), Yobst(Nobst)
    integer, intent(in) :: act(N), noiseFlag, obst_flag
    integer, intent(in) :: N, Nrod, nsteps, Nobst
    real,    intent(in) :: Rm, Rr
    real,    intent(in) :: tor, vel_act, vel_tor, dt, mR, IR, vel_noise_fact, rot_noise_fact
    real,    intent(in) :: ext_rod, cen_rod , mu
    logical, intent(in) :: reproduction
    ! shape of rod is determined by factor of size of extremes and center
    ! =======================================
    real , intent(out) :: new_XYT(N,3), new_XY_rod(Nrod,2), part_rod_forces(N,3) ! F_perf are forces in x and y and torque every particle exerted on average
    real , intent(out) :: noise(N, 3 * nsteps), vel_noise(N, nsteps), tor_noise(N, nsteps)
    ! =======================================
    real :: FX(N), FY(N), FR(N), v(N)
    real :: sig_vel_act, sig_vel_tor
    real :: F_pRX, F_pRY
    real :: F_Perp_X, F_Perp_Y, F_proj
    real :: mu_K_true = 0, mu_K =0

    real :: F_Perp(N), Friction(N)
    real :: delta_velRod, velRod_scf, velC_scf

    real :: FXrod, FYrod
    real :: torquerod, rodXcm, rodYcm, rodtheta
    real :: FXrod_eval(N), FYrod_eval(N)
    integer :: i, j, it
    real :: dx, dy, r2, drodx, drody
    real, parameter :: PI = 3.14159265358979323846264
    ! =======================================
    ! force parameters
    real :: eps = 50., epsRod=50.0, ss = 6.8, ss2, ss6, ss12, ff
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

    part_rod_forces = 0
    FXrod_eval = 0
    FYrod_eval = 0

    ! Nrod EVEN number!

    if (mod(Nrod,2) .ne. 0) then
        print*, 'Nrod is odd! Not cool!'
        stop
    endif

    ! In all simulations I did, ext_rod = cen_rod = 1 and therefore
    ! fact = 1
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

    ! DEBUG
    fact = 1

    Lrod2 = (new_XY_rod(Nrod,2)-new_XY_rod(1,2))**2 + (new_XY_rod(Nrod,1)-new_XY_rod(1,1))**2
    Lrod = sqrt(Lrod2)

    rodXcm = SUM(Xrod)/Nrod
    rodYcm = SUM(Yrod)/Nrod

    ! Diffusion for particles is determined by
    ! Rm and Rr, which creates random forces.
    ! Rm and Rr already contain the etaCol, I think.

    ! Diffusion for rod is determined by mR, which is the diffusion D0
    ! Diffusion is different in two directions:

    ! mR is a scaling of friction.
    etaCol     = etaLiq * 6 * PI * ss/2
    etaTra_par = mR * etaLiq * 2 * PI * Lrod / log(Lrod/ss)
    etaTra_per = mR * etaLiq * 4 * PI * Lrod / (log(Lrod/ss) + 1)
    etaRot     = mR * etaLiq * PI * Lrod**3 / 3. /  log(Lrod/ss)

    ! Scaling factor to be compatible with Food Project.
    eps = etaCol
    epsRod = etaCol

    ! Gaussian width for velocity distribution calculated here.
    sig_vel_act = vel_act * vel_noise_fact
    sig_vel_tor = vel_tor * rot_noise_fact

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

        rodtheta = atan2(new_XY_rod(Nrod,2)-new_XY_rod(1,2), new_XY_rod(Nrod,1)-new_XY_rod(1,1)) + 0.0000000001

        ! =============================
        ! thermal motion
        ! =============================

        ! in a reproduction step, the old noise is used again
        if (reproduction) then

            FX = oldNoise(:,1 + (it-1) * 3)
            FY = oldNoise(:,2 + (it-1) * 3)
            FR = oldNoise(:,3 + (it-1) * 3)

        else
            do i = 1, N
                FX(i) = noiseFlag * gran()*Rm*etaCol
                FY(i) = noiseFlag * gran()*Rm*etaCol
                FR(i) = noiseFlag * gran()*Rr

                ! the thermal noise is saved for reproducing this sim-step without certain particles (diff Rewards)
                noise(i, 1 + (it-1) * 3) = FX(i)
                noise(i, 2 + (it-1) * 3) = FY(i)
                noise(i, 3 + (it-1) * 3) = FR(i)

            enddo
        endif

        ! ========================================
        ! repulsion between colloids and obstacles
        ! ========================================
        if (obst_flag==1) then
            do i = 1, N
                do j = 1, Nobst
                    dx = Xobst(j) - new_XYT(i,1)
                    dy = Yobst(j) - new_XYT(i,2)
                    r2 = dx*dx + dy*dy

                    if (r2 < ss2) then
                        ff = ss12/(r2**6) - ss6/(r2**3)
                        ff = 12.*eps*ff/r2
                        ! ==== DEBUG ====
                        if (ff > 0.05 * etaCol / dt) then
                            print*, ff, sqrt(r2), 'obstacles too much'
                            ff = 0.05 * etaCol / dt
                        endif
                        ! ===============
                        FX(i) = FX(i) - ff*dx
                        FY(i) = FY(i) - ff*dy
                    endif
                enddo
            enddo
        endif

        ! ==========================
        ! repulsion between colloids
        ! ==========================

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

        F_Perp = 0.d0

        do i = 1, N

            ! ============================
            ! repulsion of single particle
            ! ============================
            F_proj = 0.d0

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

                    F_pRX = F_proj*cos(rodtheta) ! These are the forces parallel to the rod ...
                    F_pRY = F_proj*sin(rodtheta)

                    F_Perp_X = (ff*dx - F_pRX) ! ... and that is why they are subtracted here ...
                    F_Perp_Y = (ff*dy - F_pRY)

                    F_Perp(i) = F_Perp(i) + sqrt(F_Perp_X**2 + F_Perp_Y**2)

                    ! mu_K = 1       --> only perpendicular force are conserved.
                    ! mu_K = 0       --> some corrugation.
                    ! mu_K_true > 0  --> friction

                    FX(i) = FX(i) - (ff*dx - F_pRX*mu_K)
                    FY(i) = FY(i) - (ff*dy - F_pRY*mu_K)

                    FXrod = FXrod + (ff*dx - F_pRX*mu_K) ! ... and added again according to mu_K here.
                    FYrod = FYrod + (ff*dy - F_pRY*mu_K)

                    FXrod_eval(i) = 1.0/Nrod * (ff*dx - F_pRX*mu_K) ! The forces on the rod for each particle are saved, get friction corrected later
                    FYrod_eval(i) = 1.0/Nrod * (ff*dy - F_pRY*mu_K)

                    ! =======================
                    ! component of force in direction of rod
                    ! does not change torque.
                    ! =======================

                    torquerod = torquerod + (ff*dy - F_pRY*mu_K)*drodx -&
                                            (ff*dx - F_pRX*mu_K)*drody

                    part_rod_forces(i,3) =  1.0/nsteps * 1.0/Nrod * (ff*dy - F_pRY*mu_K)*drodx -&
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

                if (reproduction) then
                    vel_noise(i, it) = old_vel_noise(i, it)
                else
                    vel_noise(i, it) = noiseFlag * gran()*sig_vel_act
                endif

                v(i) = vel_act  + vel_noise(i, it)

                if (act(i)>1) then

                    if (reproduction) then
                        tor_noise(i, it) = old_tor_noise(i, it)
                    else
                        tor_noise(i, it) = noiseFlag * gran()*sig_vel_act
                    endif

                    v(i) = vel_tor + tor_noise(i, it)
                    FR(i) = FR(i) - 2*tor*(act(i)-2.5)
                endif

                FX(i) = FX(i) + cos(new_XYT(i,3))*v(i)*etaCol
                FY(i) = FY(i) + sin(new_XYT(i,3))*v(i)*etaCol

            endif

        enddo

        ! ===================================
        ! repulsion between rod and obstacles
        ! ===================================
        if (obst_flag==1) then
            do i = 1, Nobst

                F_proj = 0.d0

                do j = 1, Nrod
                    drodx = new_XY_rod(j,1) - rodXcm
                    drody = new_XY_rod(j,2) - rodYcm

                    dx = new_XY_rod(j,1) - Xobst(i)
                    dy = new_XY_rod(j,2) - Yobst(i)

                    r2 = dx*dx + dy*dy

                    if (r2 < ss2*fact(j,2)) then

                        ff = fact(j,4)*ss12/(r2**6) - fact(j,3)*ss6/(r2**3)
                        ff = 12.*epsRod*fact(j,1)*ff/r2
                        ! ==== DEBUG ====
                        if (ff > 0.05 * etaCol / dt) then
                            print*, ff, sqrt(r2), 'rod obstacles too much'
                            ff = 0.05 * etaCol / dt
                        endif
                        ! ===============
                        ! F_particle = -(ff*dx, ff*dy)
                        !rel_theta = rodtheta - atan2(ff*dy, ff*dx)

                        F_proj = (ff*dx)*cos(rodtheta) + (ff*dy)*sin(rodtheta)

                        F_pRX = F_proj*cos(rodtheta) ! These are the forces parallel to the rod ...
                        F_pRY = F_proj*sin(rodtheta)

                        F_Perp_X = (ff*dx - F_pRX) ! ... and that is why they are subtracted here ...
                        F_Perp_Y = (ff*dy - F_pRY)

                        FXrod = FXrod + ff*dx
                        FYrod = FYrod + ff*dy

                        ! =======================
                        ! component of force in direction of rod
                        ! does not change torque.
                        ! =======================

                        torquerod = torquerod + ff*dy*drodx - ff*dx*drody
                    endif
                enddo
            enddo
        endif


        ! =============================
        ! Check how much was parallel force
        ! self consistency cycle for friction

        ! F_Perp(N): already calculated. Will not change
        ! delta_velRod
        j=0
        do

            veLRod_scf = (FXrod*cos(rodtheta) + FYrod*sin(rodtheta) + SUM(Friction) ) / etaTra_par
            j = j + 1
            delta_velRod = SUM(Friction)
            Friction = 0.

            ! Self consistency for particles
            do i = 1, N
                if (mu_K_true > 0) then

                    velC_scf = (FX(i)*cos(rodtheta) + FY(i)*sin(rodtheta)) / etaCol

                    if (velC_scf .ge. velRod_scf) then
                        Friction(i) = + min(F_perp(i)*mu_K_true, &
                            abs(etaCol*(velC_scf - velRod_scf)))
                    else
                        Friction(i) = - min(F_perp(i)*mu_K_true, &
                            abs(etaCol*(velC_scf - velRod_scf)))
                    endif
                endif
            enddo

            if ( (abs(delta_velRod - SUM(Friction) ) < 1.d-3 ) .or. (j>100) ) EXIT

        enddo

        do i = 1, N
            FX(i) = FX(i) - Friction(i)*cos(rodtheta)
            FY(i) = FY(i) - Friction(i)*sin(rodtheta)

            FXrod_eval(i) = FXrod_eval(i) + Friction(i)*cos(rodtheta) ! Friction-correction of the forces for particle performance evaluation
            FYrod_eval(i) = FYrod_eval(i) + Friction(i)*sin(rodtheta)

            ! print*, FXrod_eval(i), Friction(i)*cos(rodtheta)

            part_rod_forces(i,1) = part_rod_forces(i,1) + 1.0/nsteps * FXrod_eval(i)
            part_rod_forces(i,2) = part_rod_forces(i,2) + 1.0/nsteps * FYrod_eval(i)

        enddo

        FXrod = FXrod + SUM(Friction)*cos(rodtheta)
        FYrod = FYrod + SUM(Friction)*sin(rodtheta)


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

        rodXcm = rodXcm + dt* (F_pRX/etaTra_par + (FXrod - F_pRX)/etaTra_per)
        rodYcm = rodYcm + dt* (F_pRY/etaTra_par + (FYrod - F_pRY)/etaTra_per)
        rodtheta = rodtheta + dt*torquerod/etaRot

        !print*, 'Friction', Friction, FXrod, FYrod, F_pRX, F_pRY, rodXcm, rodYcm


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


subroutine  get_o_r_rod(X, Y, Theta, Xrod, Yrod, oldXrod, oldYrod, tar_X, tar_Y, &
                        Xobst, Yobst, mode, rotDir, old_rotDir, &
                        flag_side, flag_LOS, &
                        ss, ssrod_ext, &
                        ext_rod, cen_rod, &
                        obs_type, cones, cone_angle, close_pen, prox_rew, flagFixOr, &
                        obst_flag, Nobs, N, Nrod, Nobst, Obs, Rew, touch, near2) !DEBUG
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real, intent(in)    :: X(N), Y(N), Theta(N)
    real, intent(in)    :: Xrod(Nrod), Yrod(Nrod)
    real, intent(in)    :: Xobst(Nobst), Yobst(Nobst)
    real, intent(in)    :: oldXrod(Nrod), oldYrod(Nrod), tar_X(Nrod), tar_Y(Nrod), cone_angle, close_pen, prox_rew
    integer, intent(in) :: N, Nrod, Nobs, Nobst, mode, rotDir, old_rotDir, obst_flag
    integer, intent(in) :: flag_side, obs_type, cones, flagFixOr
    logical, intent(in) :: flag_LOS
    real, intent(out)   :: Obs(N, Nobs), Rew(N), near2(N)
    integer :: i, j, k, n_cone, side
    integer :: iter_touch, adj(N,N)
    integer, intent(out) :: touch(N)
    real :: dx, dy, r, dtheta, val, th, cmRod(2), oldcmRod(2)
    real :: dx2, dy2, r2, dtheta2, dark, sp_th, ssrod, true_ss, true_ssrod
    real, intent(in) :: ss,  ssrod_ext, ext_rod, cen_rod
    real :: covered_l, covered_r, vision_l, vision_r, in_sight=0., ss_touch=6.8
    real :: dRodtheta, dRod, Rodtheta
    real :: rotRod, cone_slice, fact(Nrod)
    real, allocatable :: edge(:)
    real :: a, b, torque, rod_L, min_dist(N)
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0

    adj = 0

    min_dist = 1000

    cmRod(1) = SUM(Xrod)/Nrod + 0.0000000001
    cmRod(2) = SUM(Yrod)/Nrod + 0.0000000001

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

    oldcmRod(1) = SUM(oldXrod)/Nrod + 0.0000000001
    oldcmRod(2) = SUM(oldYrod)/Nrod + 0.0000000001

    dRod = sqrt((oldcmRod(2)-cmRod(2))**2 + (oldcmRod(1)-cmRod(1))**2 )

    Rodtheta = atan2(   Yrod(Nrod)-   Yrod(1),   Xrod(Nrod)-   Xrod(1)) + 0.0000000001
    dRodtheta = atan2(cmRod(2) - oldcmRod(2), cmRod(1) - oldcmRod(1)) + 0.0000000001

    rotRod = atan2(   Yrod(Nrod)-   Yrod(1),   Xrod(Nrod)-   Xrod(1)) - &
             atan2(oldYrod(Nrod)-oldYrod(1),oldXrod(Nrod)-oldXrod(1)) + 0.0000000001
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
        case (3)
            if (.not.(Nobs == (2+obst_flag+flag_side)*cones)) then
                print*, 'ERROR consistency  Nobs'
                print*, 'Nobs=', Nobs, ' Should be =',(2+obst_flag+flag_side)*cones
                STOP
            endif
        case (4)
            print*, 'ERROR consistency mode'
            print*, 'mode ', mode, ' is not defined'
            STOP
        case (7)
            if (.not.(Nobs == (3+obst_flag+flag_side)*cones)) then
                print*, 'ERROR consistency  Nobs. Need direction of Rod.'
                print*, 'Nobs=', Nobs, ' Should be =',(3+obst_flag+flag_side)*cones
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
            !               Yrod(1),Xrod(Nrod),Yrod(Nrod))

            side = 0
            ! side = 0 means on the same side of rod.
            ! if flag_side == 1 then visibility is across rod.
            if ((side == 0).or.(flag_side == 1)) then

                dx = X(j)-X(i)
                dy = Y(j)-Y(i)
                r = sqrt(dx*dx + dy*dy) + 0.0000000001
                ! check for adjajency and rewards
                if (r < ss_touch*1.25) then
                    adj(i,j) = 1
                    adj(j,i) = 1
                endif

                ! find the distance to the closest particle
                if (r < min_dist(i)) min_dist(i) = r

                dtheta = atan2(dy,dx) + 0.0000000001
                sp_th = atan(ss, r)/2 + 0.0000000001
                ! i to j
                ! th goes from [-pi, pi]
                th = (dtheta - Theta(i))/2./PI
                th = (th - floor(th + 0.5))*2*PI

                ! n_cone = 1 .. n_cone
                ! for theta in range [ -cone_angle , cone_angle]

                if (obs_type == 1) then
                val = min((true_ss/r),1.0)
                else if (obs_type == 2) then
                    val = min((true_ss/r**2),1.0)
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

                            dtheta2 = atan2(dy2,dx2) + 0.0000000001
                            dtheta2 = (dtheta2 - Theta(i))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2  + 0.0000000001 ! cone of shadow


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
                            dtheta2 = atan2(dy2,dx2) + 0.0000000001
                            dtheta2 = (dtheta2 - Theta(j))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2. + 0.0000000001

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

    ! seeing the rod particles and the target particles + rewards
    do i = 1, N

        do j = 1, Nrod

            ! Vision of the rod ---------------------------------------------------------------

            dx = Xrod(j)-X(i)
            dy = Yrod(j)-Y(i)
            r = sqrt(dx*dx + dy*dy) + 0.0000000001
            if (isnan(r)) then
                print*, 'Nan in r', r, i, j ! ZZZ
            endif
            if (r < near2(i)) near2(i) = r

            if (r < 0.01) then
                print*, 'very small r', r, i, j !ZZZ
            endif

            dtheta = atan2(dy,dx) + 0.0000000001
            if (isnan(dtheta)) then
                print*, 'Nan in dtheta', dtheta, i, j ! ZZZ
            endif
            ! particle sees rod
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI
            sp_th = atan(ssrod, r)/2. + 0.0000000001
            ! if (isnan(sp_th)) then
            !     print*, 'Nan in sp_th', sp_th, ssrod, i, j ! ZZZ
            ! endif
            ! -----------------------------

            if (obs_type == 1) then
                val = min((true_ssrod)/r*fact(j), 1.0)
                if (isnan(val)) then
                    print*, 'Nan in val', val, r, fact(j), true_ssrod, i, j, size(fact) ! ZZZ
                endif
            else if (obs_type == 2) then
                val = min((true_ssrod/r**2)*fact(j),1.0)
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

                        dtheta2 = atan2(dy2,dx2) + 0.0000000001
                        dtheta2 = (dtheta2 - Theta(i))/2./PI
                        dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                        dark = atan(ss, r2)/2  + 0.0000000001! cone of shadow


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

            ! End of Vision of the rod --------------------------------------------------------
            ! Vision of the target: basically the same as vision of the rod, just with different rod particles
            ! (Without near2 and touch)
            ! This does of course only apply in mode 7 (transportation problem)

            if (mode == 7) then

                dx = tar_X(j)-X(i)
                dy = tar_Y(j)-Y(i)
                r = sqrt(dx*dx + dy*dy) + 0.0000000001

                dtheta = atan2(dy,dx) + 0.0000000001
                ! particle sees rod
                th = (dtheta - Theta(i))/2./PI
                th = (th - floor(th + 0.5))*2*PI
                sp_th = atan(ssrod, r)/2. + 0.0000000001
                ! -----------------------------

                if (obs_type == 1) then
                    val = min((true_ssrod)/r*fact(j),1.0)
                else if (obs_type == 2) then
                    val = min((true_ssrod/r**2)*fact(j),1.0)
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

                            dtheta2 = atan2(dy2,dx2) + 0.0000000001
                            dtheta2 = (dtheta2 - Theta(i))/2./PI
                            dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                            dark = atan(ss, r2)/2  + 0.0000000001! cone of shadow


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

                        ! mode 7 requires more observables
                        Obs(i,n_cone+(2+flag_side)*cones) = Obs(i,n_cone+(2+flag_side)*cones)+val*in_sight
                    enddo

                endif

            endif ! end of the if mode == 7 condition for target vision

            ! End of Vision of the target -----------------------------------------------------

        enddo


        ! Vision of the obstacles: basically the same as vision of the target
        if (obst_flag==1) then
            do j = 1, Nobst

                dx = Xobst(j)-X(i)
                dy = Yobst(j)-Y(i)
                r = sqrt(dx*dx + dy*dy) + 0.0000000001

                dtheta = atan2(dy,dx) + 0.0000000001
                ! particle sees obstacle
                th = (dtheta - Theta(i))/2./PI
                th = (th - floor(th + 0.5))*2*PI
                sp_th = atan(ssrod, r)/2. + 0.0000000001
                ! -----------------------------

                if (obs_type == 1) then
                    val = min((true_ssrod)/r,1.0)
                else if (obs_type == 2) then
                    val = min((true_ssrod/r**2),1.0)
                else
                    print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                    STOP
                endif

                ! No line of sight implementation here, so no coverage
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                    ! terribly expensive way
                    ! to account for line of sight

                    vision_l = th+sp_th
                    vision_r = th-sp_th

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                        in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2.

                        ! Obstacles require another set of observables. CAREFUL WHEN INDEXING: in case of mode 3,
                        ! there are less observables (no target) than in case of mode
                        if (mode == 7) then
                            Obs(i,n_cone+(3+flag_side)*cones) = Obs(i,n_cone+(3+flag_side)*cones)+val*in_sight
                        else if (mode == 3) then
                            Obs(i,n_cone+(2+flag_side)*cones) = Obs(i,n_cone+(2+flag_side)*cones)+val*in_sight
                        endif
                    enddo
                endif
            enddo
        endif
        ! End of Vision of the obstacles -----------------------------------------------------
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
        if (isnan(torque)) then
            print*, 'NaN in torque', torque, dx, dy, a ! ZZZ
        endif


        ! different reward functions to choose from
        select case (mode)
            case (3)
                ! reward positive irrespective to direction of rotation
                if (sum(Obs(i, ((1+flag_side)*cones+(cones+1)/2):&
                               ((1+flag_side)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(rotRod, torque, touch(i), 0.0)
                    if (isnan(Rew(i))) then
                        print*, 'Nan in Rew', Rew(i), rotRod, torque, touch(i) ! ZZZ
                    endif
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                endif

            case (4)
                ! reward positive only if clockwise (-1) or anti-clockwise (+1).
                ! no penalty for translation of center of mass.
                ! NON - NEGATIVE REWARD
                if (sum(Obs(i, ((1+flag_side)*cones+(cones+1)/2):&
                               ((1+flag_side)*cones+(cones+2)/2))) > 0.) then
                    Rew(i) = reward_rotate(abs(rotRod), torque*old_rotDir, touch(i), 0.0)
                    ! Positive reward only if cooperation
                    ! Normalization: Reward is proportional to rod mass
                    Obs(i, (2+flag_side)+1) = rotDir
                endif
        end select

        Rew(i) = Rew(i) - close_pen * exp(-abs(min_dist(i))/3.6)

        Rew(i) = Rew(i) - prox_rew * (tanh((near2(i)-10*ss_touch)/10)+1)/5
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

    real FUNCTION reward_push_along(orient, dRod, dRodtheta, Rodtheta, near, cmRod, flagFixOr)
    ! Reward function for linear translation only along long axis
    implicit none
    real :: orient, dRod, dRodtheta, Rodtheta, cmRod(2)
    real :: diffT, cmRodTheta
    integer :: near, flagFixOr

    ! diffT = (Rodtheta - dRodtheta)/2./PI
    ! diffT = (diffT - floor(diffT + 0.5))*2*PI
    ! reward_push_along = near * cos( sqrt(abs(diffT))*sqrt(PI) ) * dRod

    ! MAy help to ADD PENALTY TO ROTATION:
    !1 Aaccept rotation at RRrot
    ! dRrod -> angle of rotation -> Mminimum translation = dRrod/2/pi/RRot
    ! reward_push_alogn = *max(dRrod-dRod/2/pi/RRot, 0)

    ! cos(rodtheta - dRrodtheta) >= 0 means: rod has moved at least a bit in the direction of it's orientation
    ! cos(rodtheta   -   orient) >= 0 means: particle is oriented less than perpendicular to the rod

    ! Orientation of the rod's position with respect to the origin:
    cmRodTheta = atan(cmRod(1)/cmRod(2)) - 1.571  + 0.0000000001! (subtract the initial 90°)

    if ((cos(Rodtheta - dRodtheta) >= 0) .and. (cos(Rodtheta - orient)>=0)) then

        ! Reward for particle is oriented parallelish to the rod and rod has moved in the direction of it's orientation.
        ! The last term is one if flagFixOr = 0 and cos(cmRodTheta)**3 otherwise,
        ! in order to only reward movements in the direction, the rod was originally oriented in.

        reward_push_along =  near * cos(Rodtheta - dRodtheta)**2 * dRod * (1 - (1-cos(cmRodTheta)**3)*flagFixOr)

    else if (cos(Rodtheta - orient)>=0) then

        ! No punishment, if particle is oriented parallelish to the rod
        reward_push_along = 0
    else

        ! Penalty for pushing inn the wrong direction
        reward_push_along = -near * cos(Rodtheta   -  orient)**2 * dRod
    endif

    end function reward_push_along

end subroutine