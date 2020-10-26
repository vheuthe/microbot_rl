subroutine evolve_MD(X,Y,Theta, act, Rm, Rr, dt, &
                     nsteps, tor, vel_act, vel_tor, N, new_XYT)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N)
    integer, intent(in) :: act(N)
    integer, intent(in) :: N, nsteps
    real, intent(in) :: Rm, Rr, tor, vel_act, vel_tor, dt
    ! =======================================
    real , intent(out) :: new_XYT(N,3)
    ! =======================================
    real :: velX(N), velY(N), velR(N), v
    integer :: i, j, it
    real :: dx, dy, r2
    ! =======================================
    ! force parameters
    real :: eps = 1., ss = 6.8, ss2, ss6, ss12, ff
    ! =======================================
    ss2  = ss*ss
    ss6  = ss2*ss2*ss2
    ss12 = ss6*ss6
    ! =======================================   

    new_XYT(:,1) = X
    new_XYT(:,2) = Y
    new_XYT(:,3) = Theta

    do it = 1, nsteps

        velX = 0
        velY = 0
        velR = 0

    ! =============================
    ! thermal motion + activity
    ! =============================
         
        do i = 1, N
            velX(i) = gran()*Rm
            velY(i) = gran()*Rm
            velR(i) = gran()*Rr

            if (act(i)>0) then
                ! ========================
                ! Action includes rotation
                ! ========================
                v = vel_act
                if (act(i)>1) then
                    v = vel_tor
                    velR(i) = velR(i) - 2*tor*(act(i)-2.5)
                endif
                velX(i) = velX(i) + cos(new_XYT(i,3))*v
                velY(i) = velY(i) + sin(new_XYT(i,3))*v

            endif
         enddo

    ! =============================
    ! repulsion
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
                  if (ff > 1) then
                      ff = 1
                      print*, 'too much'
                  endif
                  ! ===============
                  velX(i) = velX(i) - ff*dx
                  velY(i) = velY(i) - ff*dy
                  velX(j) = velX(j) + ff*dx
                  velY(j) = velY(j) + ff*dy                  
              endif 
          enddo
      enddo

    ! =============================
    ! move
    ! =============================

      do i = 1, N
          new_XYT(i,1) = new_XYT(i,1) + velX(i)*dt
          new_XYT(i,2) = new_XYT(i,2) + velY(i)*dt
          new_XYT(i,3) = new_XYT(i,3) + velR(i)*dt
      enddo

    enddo

    return

contains 

    real FUNCTION gran()
    !     polar form of the Box-Muller transformation
    !     http://www.taygeta.com/random/gaussian.html
      implicit none
      !  real :: rand ! using old generator
      real :: rnum(2)
      real :: x1, x2, w, y1!, y2;
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

subroutine get_neigh(X, Y, NN, N)
! ===========================================
! gets number of neighbors
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N)
    integer, intent(in) :: N
    integer, intent(out) :: NN(N,2)
    integer :: i, j, other
    real :: dx, dy, r2, ss2=8*8

    NN = 0
    
    do i = 1, N-1
        do j = i+1, N
        
            other = -int(sign(0.5, (i-N/2-0.5)*(j-N/2-0.5))-0.5)
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r2 = (dx*dx + dy*dy)
            if (r2 < ss2) then
                NN(i,other+1) = NN(i,other+1) + 1
                NN(j,other+1) = NN(j,other+1) + 1
            endif
            
        enddo
    enddo
    return

end subroutine

subroutine get_o_r_mix_tasks(X, Y, Theta, cost, mode, switch, old_switch, obs_type,&
                             cone_angle, dead_vision, flag_LOS, N, NObs, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! mode indicates whether mix (mode = 1), demix (mode = 2) or switch (mode = 3)
! switch determines wheter to mix (1 = mixing) or demix (0 = demix)
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N), cost, cone_angle, dead_vision
    integer, intent(in) :: N, NObs, switch, old_switch, mode, obs_type
    real , intent(out) :: Obs(N,NObs), Rew(N)
    integer :: i, j, k, other, n_cone, cones=-1, NN(N,2)
    real :: dx, dy, r, th, dtheta, val, sp_th
    real :: in_sight, covered_l, covered_r
    real :: vision_l, vision_r
    real :: dx2, dy2, r2, dtheta2, dark, ss=6.2, cone_slice
    real, allocatable :: edge(:,:)
    real, parameter :: PI = 3.14159265358979323846264
    logical, intent(in) :: flag_LOS

    Obs = 0
    Rew = 0
    NN = 0

    
    ! calculate real number of sight cones
    select case (mode)
    case (1) ! pure mixing
        cones = NObs / 2
    case (2) ! pure demixing
        cones = NObs / 2
    case (3) ! switch mixing/demixing
        cones = (NObs - 2) / 2
    end select
    
    ! to calculate smooth vision
    allocate(edge(cones,2))
    do i = 0, cones-1
        edge(i+1,1) = (-cone_angle/2. - (cones-1)*dead_vision/2.) + cone_angle*i/cones     + i * dead_vision
        edge(i+1,2) = (-cone_angle/2. - (cones-1)*dead_vision/2.) + cone_angle*(i+1)/cones + i * dead_vision
    enddo 
    cone_slice = cone_angle / cones
    
    ! print*, 'EDGES'
    ! do i = 1, cones
        ! print*, edge(i,1)/PI*180, edge(i,2)/PI*180
    ! enddo

    do i = 1, N-1
        do j = i+1, N
        
            other = -int(sign(0.5, (i-N/2-0.5)*(j-N/2-0.5))-0.5)
            
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            
            if (r < ss*1.5) then
                NN(i,1+other) = NN(i,1+other)+1
                NN(j,1+other) = NN(j,1+other)+1
            endif
            
            dtheta = atan2(dy,dx)
            sp_th = atan(ss, r)/2.
            ! i to j 
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI
            ! th goes from [-pi, pi]
            
            ! n_cone = 1 .. n_cone
            ! for theta in range [ -cone_angle , cone_angle]
                        
            if (obs_type == 1) then 
                val = (6.8/r)
            else if (obs_type == 2) then
                val = (6.8/r**2)
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
                    
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2. 
                    
                    Obs(i,n_cone+other*cones) = Obs(i,n_cone+other*cones)+val*in_sight
                    
                    if (Obs(i, n_cone + other*cones) > 6.8/5.5 * (cone_slice) / atan(6.2/6.) ) then
                        print*, "ERROR"
                    endif
                    
                    
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
                                        
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2. 
                    
                    Obs(j,n_cone+other*cones) = Obs(j,n_cone+other*cones)+val*in_sight
                                        
                enddo
              !    Rew(j) = Rew(j)+val*(1.-other*(1+cost))
            endif
        enddo
    enddo
    
    
    do i = 1, N
        select case (mode)
            case (1) ! pure mixing
                Rew(i) = sum(Obs(i, 1:cones)) + sum(Obs(i, (cones+1):(cones*2))) &
                    &- cost*sum(abs(Obs(i, 1:cones) - Obs(i, (cones+1):(cones*2))))
            case (2) ! pure demixing
                Rew(i) = sum(Obs(i, 1:cones)) - cost*sum(Obs(i, (cones+1):(cones*2)))
            case (3) ! switch mixing/demixing
                if (old_switch == 0) then 
                    Rew(i) = sum(Obs(i, 1:cones)) - cost*sum(Obs(i, (cones+1):(cones*2)))
                else if (old_switch == 1) then
                    Rew(i) = (sum(Obs(i, 1:cones)) + sum(Obs(i, (cones+1):(cones*2)))) &
                        &- cost*sum(abs(Obs(i, 1:cones) - Obs(i, (cones+1):(cones*2))))
                else
                    print*, 'Error: SWITCH variable not recognized.'
                    STOP
                endif
                Obs(i, (2*cones+1)+switch) = 1
            case(4) ! neighbor count
        end select
    enddo
    
    do i = 1, N
       if ( sum(Obs(i,1:(2*cones))) .eq. 0 ) Rew(i) = -2
    enddo
    
    return

end subroutine

subroutine get_o_r_food_task(X, Y, Theta, obs_type, cone_angle, dead_vision, &
                                        ratio_rew, XP, YP, Food, N, NObs, Obs, Rew, Eaten)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    ! input
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N)
    integer, intent(in) :: obs_type
    real, intent(in) :: cone_angle, dead_vision
    real, intent(in) :: ratio_rew, XP, YP, Food
    integer, intent(in) :: N, NObs
    ! output
    real , intent(out) :: Obs(N,NObs), Rew(N), Eaten
    ! internal processes
    integer :: i, j, k, n_cone, cones=-1
    real :: dx, dy, r, dtheta, val, th, th_orient
    real :: in_sight, covered_l, covered_r
    real :: vision_l, vision_r
    real :: dx2, dy2, r2, dtheta2, dark, ss=6.2, sp_th, cone_slice
    real :: max_payoff
    real :: food_width
    real, allocatable :: edge(:,:)
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
    
    ! CONES
    ! 2/3 orientational weighted vision - 1/3 food vision
    cones = NObs / 3

    ! to calculate smooth vision
    allocate(edge(cones,2))
    do i = 0, cones-1
        edge(i+1,1) = (-cone_angle/2. - (cones-1)*dead_vision/2.) + cone_angle*i/cones     + i * dead_vision
        edge(i+1,2) = (-cone_angle/2. - (cones-1)*dead_vision/2.) + cone_angle*(i+1)/cones + i * dead_vision
    enddo 
    cone_slice = edge(0,2) - edge(0,1)

    ! Maximum payoff for compactness
    if (obs_type == 1) then 
        max_payoff = 0.75 * (6.8 / ss)    * (1 - ratio_rew) * (cone_angle / 2.) / atan(1.0) 
    else if (obs_type == 2) then
        max_payoff = 0.75 * (6.8 / ss)**2 * (1 - ratio_rew) * (cone_angle / 2.) / atan(1.0) 
    else 
        print*, 'ERROR NO OBS_TYPE IS DEFINED!'
        STOP
    endif

    do i = 1, N-1
        
        ! FELLOW PARTICLES
        do j = i+1, N
        
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            
            dtheta = atan2(dy,dx)
            sp_th = atan(ss, r)/2.
            ! i to j ============================================
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI 
            ! th in [-pi, pi]
            th_orient = (Theta(j)- Theta(i))/2./PI
            th_orient = (th_orient - floor(th_orient + 0.5))*2*PI 
            
            if (obs_type == 1) then 
                val = (6.8/r)
            else if (obs_type == 2) then
                val = (6.8/r)**2
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
            
            ! penalty for touching
            if (r < 13.6) then ! 2 x diameter
                Rew(i) = Rew(i) + 0.5*(tanh((r-6.8)/2)-1)*10 ! penalty to touch
                Rew(j) = Rew(j) + 0.5*(tanh((r-6.8)/2)-1)*10 ! penalty to touch
            endif
            
            
            covered_l = 0
            covered_r = 0   
            
            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
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
                
                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2. 
                    
                    Obs(i,2*n_cone-1) = Obs(i,2*n_cone-1)+val*in_sight*cos(th_orient)
                    Obs(i,2*n_cone  ) = Obs(i,2*n_cone  )+val*in_sight*sin(th_orient)
                    Rew(i) = Rew(i) +  val*in_sight * (1-ratio_rew)                     
                    
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

            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
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

                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r

                do n_cone= 1, cones
                    ! fraction of particle in sight
                    ! if particle in cone
                    in_sight = 0.
                    in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) /sp_th/2.                     

                    Obs(j,2*n_cone-1) = Obs(j,2*n_cone-1)+val*in_sight*cos(th_orient)
                    Obs(j,2*n_cone  ) = Obs(j,2*n_cone  )+val*in_sight*sin(th_orient)
                    
                    Rew(j) = Rew(j) +  val*in_sight * (1-ratio_rew)                     
                enddo
            endif
        enddo
    enddo

    do i = 1, N
        Rew(i) = min(max_payoff, Rew(i))
    enddo

    food_width = sqrt(Food) ! Food is input
    
    if (food_width > 0) then
        do i = 1, N-1        
            ! FOOD SOURCE
            dx = XP - X(i)
            dy = YP - Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            sp_th = atan(food_width, r)/2.
            ! i to j 
            th = (dtheta - Theta(i))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = (th - floor(th + 0.5))*2*PI
            val = food_width / r
            
            ! VALUE OF FOOD
            if (r > food_width) then
                covered_l = 0
                covered_r = 0   
                
                if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
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
                    
                    vision_l = th+sp_th-covered_l
                    vision_r = th-sp_th+covered_r

                    do n_cone= 1, cones
                        ! fraction of particle in sight
                        ! if particle in cone
                        in_sight = 0.
                        in_sight = max((min(vision_l, edge(n_cone,2)) - max(vision_r, edge(n_cone,1))), 0.) / sp_th / 2. 
                        
                        Obs(i,n_cone + 2*cones) = Obs(i,n_cone + 2*cones)+val*in_sight
                    enddo


                endif
            else
                Obs(i,(2*cones+1):3*cones) = cone_slice
                Rew(i) = Rew(i) + 1*ratio_rew
                Eaten = Eaten + 1
            endif
        enddo
    endif

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