subroutine evolve_MD(XYT, act, nsteps, dt, torque, torque_noise, &
                     vel_act, vel_tor, vel_noise, &
                     N, new_XYT)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real , intent(in) :: XYT(N,3)
    integer, intent(in) :: act(N)
    integer, intent(in) :: N, nsteps
    real, intent(in) :: dt, torque, torque_noise
    real, intent(in) :: vel_act, vel_tor, vel_noise
    ! =======================================
    real , intent(out) :: new_XYT(N,3)
    ! =======================================
    real :: velX(N), velY(N), velR(N), v
    integer :: i, j, it
    real :: dx, dy, r2
    ! =======================================
    ! force parameters
    real :: eps = 1., size = 6.2, size2, size6, size12, ff
    ! =======================================
    size2  = size*size
    size6  = size2*size2*size2
    size12 = size6*size6
    ! =======================================

    new_XYT = XYT

    do it = 1, nsteps

        velX = 0
        velY = 0
        velR = 0

        ! we neglect diffusion as its small compared to noise on propulsion

        ! active propulsion
        do i = 1, N
            if (act(i)>0) then ! not passive
                if (act(i)>1) then ! rotating
                    v = vel_tor + gran() * vel_noise * vel_tor
                    velR(i) = -2*(act(i)-2.5) * torque * (1 + gran()*torque_noise)
                else
                    v = vel_act + gran() * vel_noise * vel_act
                endif
                velX(i) = cos(new_XYT(i,3)) * v
                velY(i) = sin(new_XYT(i,3)) * v

            endif
        enddo

        ! particle-particle repulsion
        do i = 1, N-1
            do j = i+1, N
                dx = new_XYT(j,1) - new_XYT(i,1)
                dy = new_XYT(j,2) - new_XYT(i,2)
                r2 = dx*dx + dy*dy
                if (r2 < size2) then
                    ff = size12/(r2**6) - size6/(r2**3)
                    ff = 12.*eps*ff/r2
                    ! ==== DEBUG ====
                    if (ff > 1) then
                        print *, 'too much: ', ff
                        ff = 1
                    endif
                    ! ===============
                    velX(i) = velX(i) - ff*dx
                    velY(i) = velY(i) - ff*dy
                    velX(j) = velX(j) + ff*dx
                    velY(j) = velY(j) + ff*dy
                endif
            enddo
        enddo

        ! move
        do i = 1, N
            new_XYT(i,1) = new_XYT(i,1) + velX(i)*dt
            new_XYT(i,2) = new_XYT(i,2) + velY(i)*dt
            new_XYT(i,3) = new_XYT(i,3) + velR(i)*dt
        enddo

    enddo

    return

contains

    real function gran()
    !     polar form of the Box-Muller transformation
    !     http://www.taygeta.com/random/gaussian.html
        implicit none
        real :: rnum(2)
        real :: x1, x2, w, y1!, y2;
        do
            call random_number(rnum)
            x1 = 2.0 * rnum(1) - 1.0
            x2 = 2.0 * rnum(2) - 1.0
            w = x1 * x1 + x2 * x2
            if( w < 1.0 ) exit
        enddo
        w = sqrt( (-2.0 * log( w ) ) / w )
        y1 = x1 * w
        !  y2 = x2 * w ! there is here a second independent random number for free
        gran = y1
        return
    end function gran

end subroutine

subroutine get_neigh(X, Y, NN, N)
! ===========================================
! gets number of neighbors
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N)
    integer, intent(in) :: N
    integer, intent(out) :: NN(N,2)
    integer :: i, j, other
    real :: dx, dy, r2, size2=6.2**2

    NN = 0

    do i = 1, N-1
        do j = i+1, N

            other = -int(sign(0.5, (i-N/2-0.5)*(j-N/2-0.5))-0.5)
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r2 = (dx*dx + dy*dy)
            if (r2 < size2) then
                NN(i,other+1) = NN(i,other+1) + 1
                NN(j,other+1) = NN(j,other+1) + 1
            endif

        enddo
    enddo
    return

end subroutine

subroutine get_o_r_food_task(X, Y, Theta, XFood, YFood, RFood, &
               vision_angle, cones, dead_vision, obs_type, phys_size, vis_size, &
               food_rew, nn_rew_cones, max_nn_rew, tp_type, touch_penalty, &
               N, NFood, Obs, Rew, Eaten)

    ! ===========================================
    ! gets observables and rewards from positions
    ! ===========================================
    implicit none

    ! Particles and (multiple) food sources
    real , intent(in) :: X(N), Y(N), Theta(N)
    real, intent(in) :: XFood(NFood), YFood(NFood), RFood(NFood)
    ! observables
    integer, intent(in) :: cones, obs_type
    real, intent(in) :: vision_angle, dead_vision, phys_size, vis_size
    ! rewards
    integer, intent(in) :: nn_rew_cones, tp_type
    real, intent(in) :: food_rew, max_nn_rew, touch_penalty
    ! implicit input (infered by f2py)
    integer, intent(in) :: N, NFood
    ! output
    real , intent(out) :: Obs(N,4*cones), Rew(N), Eaten(NFood)

    ! internal processes
    integer :: i, j, k, n_cone, start_cone, iFood
    real :: dx, dy, r, dtheta, val, th, th_orient
    real :: in_sight, covered_l, covered_r
    real :: vision_l, vision_r
    real :: dx2, dy2, r2, dtheta2, dark, sp_th, cone_slice

    ! vision of food
    integer :: bins = 200
    real :: food_angle, max_rew, nn_rew
    real, allocatable :: edge(:,:), food_sight(:)
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0


    ! to calculate smooth vision
    allocate(edge(cones,2))
    allocate(food_sight(-bins/2:(bins/2-1)))
    do i = 0, cones-1
        edge(i+1,1) = (-vision_angle/2. - (cones-1)*dead_vision/2.) + vision_angle*i/cones     + i * dead_vision
        edge(i+1,2) = (-vision_angle/2. - (cones-1)*dead_vision/2.) + vision_angle*(i+1)/cones + i * dead_vision
    enddo
    cone_slice = edge(1,2) - edge(1,1)

    ! calculate prefactor of 0 cones nn_rew and touch penalty
    ! included factors are:
    ! "(1-food_rew)" to have a ratio of 1:1 at 50% food_rew
    ! "2 / sqrt(27 * touch_penalty)" the algebraic maximum of 1/r-TP/r³ at r > 1
    ! "10 * sqrt(N)" an emperical factor to account for number of neighbors and relative food prescence per time
    nn_rew = (1 - food_rew) / min(1.0, sqrt(4.0 / 27 / touch_penalty)) / (10 * sqrt(1.0 * N))

    do i = 1, N-1

        ! FELLOW PARTICLES
        do j = i+1, N

            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)

            dtheta = atan2(dy,dx)
            sp_th = asin(vis_size/2. / r)
            ! i to j ============================================
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI
            ! th in [-pi, pi]
            th_orient = (Theta(j)- Theta(i))/2./PI
            th_orient = (th_orient - floor(th_orient + 0.5))*2*PI

            if (obs_type == 1) then
                val = min((phys_size/r), 1.0)
            else if (obs_type == 2) then
                val = min((phys_size/r)**2, 1.0)
            else
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif

            ! penalty for touching
            if (r < 13.6) then ! 2 x diameter
                if (tp_type == 1) then
                    Rew(i) = Rew(i) + 0.5*(tanh((r-phys_size)/2)-1)*touch_penalty * (1 - food_rew) !, penalty to touch - SCALED!
                    Rew(j) = Rew(j) + 0.5*(tanh((r-phys_size)/2)-1)*touch_penalty * (1 - food_rew) !, penalty to touch - SCALED!
                else if (tp_type == 2) then
                    ! only penalize for the closest neighbor
                    Rew(i) = min(Rew(i), 0.5*(tanh((r-phys_size)/2)-1)*touch_penalty * (1 - food_rew))
                    Rew(j) = min(Rew(j), 0.5*(tanh((r-phys_size)/2)-1)*touch_penalty * (1 - food_rew))
                endif
            endif
            if (tp_type == 3) then
                ! penalize all with 1/r^3
                Rew(i) = Rew(i) - nn_rew * touch_penalty * min((phys_size/r), 1.0)**3
                Rew(j) = Rew(j) - nn_rew * touch_penalty * min((phys_size/r), 1.0)**3
            endif

            ! handle observable independent reward here
            if (nn_rew_cones == 0) then
                ! reward all with 1/r
                Rew(i) = Rew(i) + nn_rew * min((phys_size/r), 1.0)
                Rew(j) = Rew(j) + nn_rew * min((phys_size/r), 1.0)
            endif

            covered_l = 0
            covered_r = 0

            if ((th>-(vision_angle/2.+sp_th)).and.(th<(vision_angle/2.+sp_th))) then
            ! terribly expensive way
            ! to account for line of sight
                do k = 1, N

                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(i)
                    dy2 = Y(k)-Y(i)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)

                    if (r2 > r) cycle !only closer particles can obscure

                    dtheta2 = atan2(dy2,dx2)
                    dtheta2 = (dtheta2 - Theta(i))/2./PI
                    dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                    dark = asin(vis_size / 2. / r2) ! cone of shadow

                    if (abs(th-dtheta2) < dark + sp_th) then
                        if (th .lt. dtheta2) then
                            covered_l = max(covered_l, (th + sp_th) - (dtheta2 - dark))
                        else if (th .ge. dtheta2) then
                            covered_r = max(covered_r, (dtheta2 + dark) - (th - sp_th))
                        endif

                        if (covered_l+covered_r > 2*sp_th) exit  ! fully covered
                    endif
                enddo

                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2.

                    Obs(i,3*n_cone-2) = Obs(i,3*n_cone-2)+val*in_sight
                    Obs(i,3*n_cone-1) = Obs(i,3*n_cone-1)+val*in_sight*cos(th_orient)
                    Obs(i,3*n_cone  ) = Obs(i,3*n_cone  )+val*in_sight*sin(th_orient)
                    !Rew(i) = Rew(i) +  val*in_sight * (1-food_rew)

                enddo
            endif


            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = (th - floor(th + 0.5))*2*PI
            th_orient = (Theta(i)- Theta(j))/2./PI
            th_orient = (th_orient - floor(th_orient + 0.5))*2*PI

            covered_l = 0
            covered_r = 0

            if ((th>-(vision_angle/2.+sp_th)).and.(th<(vision_angle/2.+sp_th))) then
            ! terribly expensive way
            ! to account for line of sight
                do k = 1, N
                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(j)
                    dy2 = Y(k)-Y(j)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)
                    if (r2 > r) cycle
                    dtheta2 = atan2(dy2,dx2)
                    dtheta2 = (dtheta2 - Theta(j))/2./PI
                    dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                    dark = asin(vis_size / 2. / r2)

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

                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2.

                    Obs(j,3*n_cone-2) = Obs(j,3*n_cone-2)+val*in_sight
                    Obs(j,3*n_cone-1) = Obs(j,3*n_cone-1)+val*in_sight*cos(th_orient)
                    Obs(j,3*n_cone  ) = Obs(j,3*n_cone  )+val*in_sight*sin(th_orient)

                    !Rew(j) = Rew(j) +  val*in_sight * (1-food_rew)
                enddo
            endif
        enddo
    enddo


    if (nn_rew_cones == 0) then
        ! handled above
    else if (nn_rew_cones == 2) then
        do i = 1, N
            max_rew = -1
            do j = 1, cones-1
                if (Obs(i,3*j-2) + Obs(i,3*(j+1)-2) > max_rew) max_rew = Obs(i,3*j-2) + Obs(i,3*(j+1)-2)
            enddo
            Rew(i) = Rew(i) + max_rew * (1-food_rew)
            Rew(i) = min(max_nn_rew * (1-food_rew), Rew(i))
        enddo
    else if (nn_rew_cones == cones) then
        do i = 1, N
            do j = 1, cones
                Rew(i) = Rew(i) + Obs(i,3*j-2) * (1-food_rew)
            enddo
            Rew(i) = min(max_nn_rew * (1-food_rew), Rew(i))
        enddo
    else
        print*, 'ERROR: unimplemented number of rewarded cones'
        stop
    endif


    food_loop: do iFood = 1, NFood

        if (RFood(iFood) > 0) then
            particle_loop: do i = 1, N
                ! FOOD SOURCE
                dx = XFood(iFood) - X(i)
                dy = YFood(iFood) - Y(i)
                r = sqrt(dx*dx + dy*dy)
                dtheta = atan2(dy,dx)
                sp_th = asin(RFood(iFood) / r)
                ! i to j
                th = (dtheta - Theta(i))/2./PI
                ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
                th = (th - floor(th + 0.5))*2*PI

                ! Value of food,
                ! factor 3 is there to make it comparable in amount with particle obs
                if (obs_type == 1) then
                    val = 3 * min((RFood(iFood)/r), 1.0)
                else if (obs_type == 2) then
                    val = 3 * min((RFood(iFood)/r)**2, 1.0)
                else
                    print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                    STOP
                endif

                ! VALUE OF FOOD
                if (r > RFood(iFood)) then

                    food_sight = 0
                    food_angle = sp_th/(bins/2)

                    ! VISION OF FOOD IS DIVIDED IN DISCRETE BINS.
                    do j = -bins/2, bins/2-1
                        if ((th + (j+0.5)*food_angle > -(vision_angle/2.)).and.&
                            (th + (j+0.5)*food_angle <  (vision_angle/2.))) then
                            food_sight(j) = 1
                        endif
                    enddo

                    if (all(food_sight == 0)) cycle particle_loop

                    ! even more terribly expensive way
                    ! to account for line of sight
                    LOS_loop: do k = 1, N

                        if (i==k) cycle
                        dx2 = X(k)-X(i)
                        dy2 = Y(k)-Y(i)
                        r2 = sqrt(dx2*dx2 + dy2*dy2)

                        if ( r2 > r) cycle !only closer particles can obscure

                        dtheta2 = atan2(dy2,dx2)
                        dtheta2 = (dtheta2 - Theta(i))/2./PI
                        dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                        dark = asin(vis_size / 2. / r2) ! cone of shadow

                        if (abs(th-dtheta2) < dark + sp_th) then
                            do j = -bins/2, bins/2-1
                                if ((th + (j+0.5)*food_angle > (dtheta2 - dark)).and.&
                                    (th + (j+0.5)*food_angle < (dtheta2 + dark))) then
                                    food_sight(j) = 0
                                endif
                            enddo

                            if (all(food_sight == 0)) cycle particle_loop  ! fully covered
                        endif
                    enddo LOS_loop

                    do j = -bins/2, bins/2-1
                        start_cone = 1
                        cone_loop: do n_cone = start_cone, cones
                            if ((th + (j+0.5)*food_angle < edge(n_cone,2)).and.&
                                (th + (j+0.5)*food_angle >= edge(n_cone,1))) then
                                Obs(i, n_cone + 3*cones) = Obs(i, n_cone + 3*cones) + val * 1. / bins * food_sight(j)
                                start_cone = n_cone
                                exit cone_loop
                            endif
                        enddo cone_loop
                    enddo

                else
                    Obs(i,(3*cones+1):4*cones) = Obs(i,(3*cones+1):4*cones) + cone_slice / vision_angle * val
                    Rew(i) = Rew(i) + 1*food_rew
                    Eaten(iFood) = Eaten(iFood) + 1
                endif

            enddo particle_loop
        endif
    enddo food_loop

    return

end subroutine

subroutine get_order_param(X, Y, Theta, order_gl, swirl_gl, N)
    ! ===========================================
    ! gets two order parameters:
    ! swirl and order
    ! ===========================================
    ! input
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N)
    integer, intent(in) :: N
    ! output
    real , intent(out) :: order_gl, swirl_gl !, order_loc(N), swirl_loc(N)
    ! internal processes
    integer :: i
    real :: dx, dy, r, or_x, or_y
    real :: Xcom, Ycom
    real, parameter :: PI = 3.14159265358979323846264


    Xcom = 1./N * sum(X)
    Ycom = 1./N * sum(Y)

    swirl_gl = 0
    order_gl = 0
    or_x = 0
    or_y = 0

    do i=1,N
        dx = X(i) - Xcom
        dy = Y(i) - Ycom
        r = sqrt(dx**2 + dy**2)
        swirl_gl = swirl_gl + (dx*sin(Theta(i)) - dy*cos(Theta(i)) ) / r
        or_x = or_x + cos(Theta(i))
        or_y = or_y + sin(Theta(i))
    enddo

    order_gl = sqrt(or_x**2 + or_y**2) / N
    swirl_gl = swirl_gl / N

end subroutine