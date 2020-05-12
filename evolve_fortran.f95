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
      real :: x1, x2, w, y1, y2;
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

subroutine get_o_r_mix_tasks(X, Y, Theta, cost, mode, switch, N, NObs, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! mode indicates whether mix (mode = 1), demix (mode = 2) or switch (mode = 3)
! switch determines wheter to mix (1 = mixing) or demix (0 = demix)
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N), cost
    integer, intent(in) :: N, NObs, switch, mode
    real , intent(out) :: Obs(N,NObs), Rew(N)
    integer :: i, j, other, n_cone
    real :: dx, dy, r, dtheta, val, th
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
    do i = 1, N-1
        do j = i+1, N
        
            other = -int(sign(0.5, (i-N/2-0.5)*(j-N/2-0.5))-0.5)
            
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            ! i to j 
            th = (dtheta - Theta(i))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            val = (6.8/r)
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(i,n_cone+other*5) = Obs(i,n_cone+other*5)+val
            !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
            endif
            
            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(j,n_cone+other*5) = Obs(j,n_cone+other*5)+val
            !    Rew(j) = Rew(j)+val*(1.-other*(1+cost))
            endif

        enddo
    enddo
    
    
    do i = 1, N
        select case (mode)
            case (1) ! pure mixing
                Rew(i) = sum(Obs(i, 1:5)) + cost*sum(Obs(i, 6:10)) &
                  &- abs(sum(Obs(i, 1:5)) - cost*sum(Obs(i, 6:10)))
            case (2) ! pure demixing
                Rew(i) = sum(Obs(i, 1:5)) - cost*sum(Obs(i, 6:10))
            case (3) ! switch mixing/demixing
                if (switch == 0) then 
                    Rew(i) = sum(Obs(i, 1:5)) - cost*sum(Obs(i, 6:10))
                else if (switch == 1) then
                    Rew(i) = sum(Obs(i, 1:5)) + cost*sum(Obs(i, 6:10)) &
                      &- abs(sum(Obs(i, 1:5)) - cost*sum(Obs(i, 6:10)))
                else
                    print*, 'Error: SWITCH variable not recognized.'
                    STOP
                endif
                Obs(i, 11) = switch
        end select
    enddo
    
    do i = 1, N
       if ( sum(Obs(i,:)) .eq. 0 ) Rew(i) = -2
    enddo
    
    return

end subroutine


subroutine get_o_r_group_task(X,Y,Theta,N,Obs,Rew)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N)
    integer, intent(in) :: N
    real , intent(out) :: Obs(N,5), Rew(N)
    integer :: i, j, n_cone
    real :: dx, dy, r, dtheta, val, th
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
    do i = 1, N-1
        do j = i+1, N
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)

            ! i to j 
            th = (dtheta - Theta(i))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            val = (6.8/r)
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(i,n_cone) = Obs(i,n_cone)+val
                Rew(i) = Rew(i)+val
            endif
            
            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(j,n_cone) = Obs(j,n_cone)+val
                Rew(j) = Rew(j)+val
            endif
        enddo
    enddo

    do i = 1, N
       if ( sum(Obs(i,:)) .eq. 0 ) Rew(i) = -2
    enddo

    return

end subroutine

