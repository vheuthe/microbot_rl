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

subroutine get_o_r_mix_tasks(X, Y, Theta, cost, mode, switch, obs_type, cone_angle, N, NObs, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! mode indicates whether mix (mode = 1), demix (mode = 2) or switch (mode = 3)
! switch determines wheter to mix (1 = mixing) or demix (0 = demix)
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N), cost, cone_angle
    integer, intent(in) :: N, NObs, switch, mode, obs_type
    real , intent(out) :: Obs(N,NObs), Rew(N)
    integer :: i, j, k, other, n_cone, cones=-1, visible
    real :: dx, dy, r, th, dtheta, val, cone_angle_reduced
    real :: dx2, dy2, r2, dtheta2, dark, ss=6.2
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
    
    ! cone_angle must be a positive angle in radiants
    cone_angle_reduced = cone_angle / 2. / PI
    
    ! calculate real number of sight cones
    select case (mode)
    case (1) ! pure mixing
        cones = NObs / 2
    case (2) ! pure demixing
        cones = NObs / 2
    case (3) ! switch mixing/demixing
        cones = (NObs - 2) / 2
    end select
    
    do i = 1, N-1
        do j = i+1, N
        
            other = -int(sign(0.5, (i-N/2-0.5)*(j-N/2-0.5))-0.5)
            
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            ! i to j 
            th = (dtheta - Theta(i))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = th - floor(th + 0.5)

            ! n_cone = 1 .. n_cone
            ! for theta in range [ -cone_angle , cone_angle]
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
            
            if (obs_type == 1) then 
                val = (6.8/r)
            else if (obs_type == 2) then
                val = (6.8/r**2)
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
            
            if ((n_cone < cones+1) .and. (n_cone>0)) then
                ! terribly expensive way
                ! to account for line of sight
                visible = 1
                do k = 1, N 
                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(i)
                    dy2 = Y(k)-Y(i)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)
                    if (r2 > r) cycle
                    dtheta2 = atan2(dy2,dx2)
                    dark = atan(ss, r2)
                    if (abs(dtheta-dtheta2) < dark) then
                        visible = 0
                        exit
                    endif
                enddo
                Obs(i,n_cone+other*cones) = Obs(i,n_cone+other*cones)+val*visible
            !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
            endif
            
            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = th - floor(th + 0.5)

            ! n_cone = 1 .. n_cone
            ! for theta in range [ -cone_angle , cone_angle]
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
            
            if ((n_cone < cones+1) .and. (n_cone>0)) then
                ! terribly expensive way
                ! to account for line of sight
                visible = 1
                dtheta = atan2(-dy,-dx)
                do k = 1, N 
                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(j)
                    dy2 = Y(k)-Y(j)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)
                    if (r2 > r) cycle
                    dtheta2 = atan2(dy2,dx2)
                    dark = atan(ss, r2)
                    if (abs(dtheta-dtheta2)<dark) then
                        visible = 0
                        exit
                    endif
                enddo            

                Obs(j,n_cone+other*cones) = Obs(j,n_cone+other*cones)+val*visible
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
                if (switch == 0) then 
                    Rew(i) = sum(Obs(i, 1:cones)) - cost*sum(Obs(i, (cones+1):(cones*2)))
                else if (switch == 1) then
                    Rew(i) = (sum(Obs(i, 1:cones)) + sum(Obs(i, (cones+1):(cones*2)))) &
                        &- cost*sum(abs(Obs(i, 1:cones) - Obs(i, (cones+1):(cones*2))))
                else
                    print*, 'Error: SWITCH variable not recognized.'
                    STOP
                endif
                Obs(i, (2*cones+1)+switch) = 1
        end select
    enddo
    
    do i = 1, N
       if ( sum(Obs(i,1:(2*cones))) .eq. 0 ) Rew(i) = -2
    enddo
    
    return

end subroutine


subroutine get_o_r_group_predator_task(X, Y, Theta, obs_type, cone_angle, flag_P, XP, YP, N, NObs, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N), XP, YP, cone_angle
    logical, intent(in) :: flag_P
    integer, intent(in) :: N, NObs, obs_type
    real , intent(out) :: Obs(N,NObs), Rew(N)
    integer :: i, j, k, n_cone, cones=-1, visible
    real :: dx, dy, r, dtheta, val, th, cone_angle_reduced
    real :: dx2, dy2, r2, dtheta2, dark, ss=6.2

    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
    
    ! cone_angle must be a positive angle in radiants
    cone_angle_reduced = cone_angle / 2. / PI
    
    ! calculate real number of sight cones
    cones = NObs
    if (flag_P) cones = NObs / 2
   
    Obs = 0
    Rew = 0

    do i = 1, N-1
        
        ! FELLOW PARTICLES
        do j = i+1, N
        
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            
            ! i to j ============================================
            th = (dtheta - Theta(i))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = th - floor(th + 0.5)

            ! n_cone = 1 .. n_cone
            ! for theta in range [ -cone_angle , cone_angle]
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced)*cones )+1
                        
            if (obs_type == 1) then 
                val = (6.8/r)
            else if (obs_type == 2) then
                val = (6.8/r**2)
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
            
            if ((n_cone < cones+1) .and. (n_cone>0)) then
                ! terribly expensive way
                ! to account for line of sight
                visible = 1
                do k = 1, N 
                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(i)
                    dy2 = Y(k)-Y(i)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)
                    if (r2 > r) cycle
                    dtheta2 = atan2(dy2,dx2)
                    dark = atan(ss, r2)
                    if (abs(dtheta-dtheta2) < dark) then
                        visible = 0
                        exit
                    endif
                enddo
                Obs(i,n_cone) = Obs(i,n_cone)+val*visible
            !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
            endif
            
            ! j to i ============================================
            th = (dtheta + PI - Theta(j))/2./PI
            ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
            th = th - floor(th + 0.5)

            ! n_cone = 1 .. n_cone
            ! for theta in range [ -cone_angle , cone_angle]
            n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
            
            if ((n_cone < cones+1) .and. (n_cone>0)) then
                ! terribly expensive way
                ! to account for line of sight
                dtheta = atan2(-dy,-dx)
                visible = 0
                do k = 1, N 
                    if ((i==k).or.(j==k)) cycle
                    dx2 = X(k)-X(j)
                    dy2 = Y(k)-Y(j)
                    r2 = sqrt(dx2*dx2 + dy2*dy2)
                    if (r2 > r) cycle
                    dtheta2 = atan2(dy2,dx2)
                    dark = atan(ss, r2)
                    if (abs(dtheta-dtheta2)<dark) then
                        visible = 0
                        exit
                    endif
                enddo            

                Obs(j,n_cone) = Obs(j,n_cone)+val
            !    Rew(j) = Rew(j)+val*(1.-other*(1+cost))
            endif
        enddo
        
        ! PREDATOR
        dx = X(j)-XP
        dy = Y(j)-YP
        r = sqrt(dx*dx + dy*dy)
        dtheta = atan2(dy,dx)
        ! i to j 
        th = (dtheta - Theta(i))/2./PI
        ! th goes from [-0.5, 0.5], correspondin to [-pi, pi]
        th = th - floor(th + 0.5)

        ! n_cone = 1 .. n_cone
        ! for theta in range [ -cone_angle , cone_angle]
        n_cone = floor( (th + cone_angle_reduced)/(2.*cone_angle_reduced) * cones )+1
        
        if (obs_type == 1) then 
            val = (6.8/r)
        else if (obs_type == 2) then
            val = (6.8/r)**2 / 6.8
        else 
            print*, 'ERROR NO OBS_TYPE IS DEFINED!'
            STOP
        endif
        
        if ((n_cone < cones+1) .and. (n_cone>0)) then
            ! terribly expensive way
            ! to account for line of sight
            visible = 1
            do k = 1, N 
                if ((i==k).or.(j==k)) cycle
                dx2 = X(k)-X(i)
                dy2 = Y(k)-Y(i)
                r2 = sqrt(dx2*dx2 + dy2*dy2)
                if (r2 > r) cycle
                dtheta2 = atan2(dy2,dx2)
                dark = atan(ss, r2)
                if (abs(dtheta-dtheta2) < dark) then
                    visible = 0
                    exit
                endif
            enddo
            Obs(i,n_cone+cones) = Obs(i,n_cone+cones)+val*visible
        !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
        endif
        Rew(i) = -10*(6.8/r)**3
    enddo

    do i = 1, N
       if ( sum(Obs(i,:cones)) .eq. 0 ) Rew(i) = -2
    enddo

    return

end subroutine

