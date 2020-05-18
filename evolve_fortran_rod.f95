subroutine evolve_md_rod(mR, X,Y,Theta, Xrod, Yrod, act, Rm, Rr, dt, &
                     nsteps, tor, vel_act, vel_tor, N, Nrod, new_XYT, new_XY_rod)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real,    intent(in) :: X(N), Y(N), Theta(N)
    real,    intent(in) :: Xrod(Nrod), Yrod(Nrod)
    integer, intent(in) :: act(N)
    integer, intent(in) :: N, Nrod, nsteps
    real,    intent(in) :: Rm, Rr, tor, vel_act, vel_tor, dt, mR
    ! =======================================
    real , intent(out) :: new_XYT(N,3), new_XY_rod(Nrod,2)
    ! =======================================
    real :: velX(N), velY(N), velR(N), v
    real :: velXrod, velYrod, torquerod, rodXcm, rodYcm, rodtheta
    integer :: i, j, it
    real :: dx, dy, r2, drodx, drody
    ! =======================================
    ! force parameters
    real :: eps = 1., ss = 4.8, ss2, ss6, ss12, ff, epsRod=1.0
    ! =======================================
    ! rod parameters
    real :: Irod = 0.d0, Lrod2 = 0.d0, massRod
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
    massRod = mR / Nrod

    rodXcm = SUM(new_XY_rod(:,1))/Nrod
    rodYcm = SUM(new_XY_rod(:,2))/Nrod
    rodtheta = atan2(new_XY_rod(Nrod,2)-new_XY_rod(1,2), new_XY_rod(Nrod,1)-new_XY_rod(1,1))

    Lrod2 = (new_XY_rod(Nrod,2)-new_XY_rod(1,2))**2 + (new_XY_rod(Nrod,1)-new_XY_rod(1,1))**2
    Irod = 1. / 12. * massRod * Nrod * Lrod2

    do it = 1, nsteps

        velX = 0.d0
        velY = 0.d0
        velR = 0.d0

        velXrod = 0.d0
        velYrod = 0.d0
        torquerod = 0.d0


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
    ! repulsion with rod
    ! =============================

     do j = 1, Nrod
         drodx = new_XY_rod(j,1) - rodXcm
         drody = new_XY_rod(j,2) - rodYcm

         do i = 1, N
              dx = new_XY_rod(j,1) - new_XYT(i,1)
              dy = new_XY_rod(j,2) - new_XYT(i,2)
              r2 = dx*dx + dy*dy
              if (r2 < ss2) then
                  ff = ss12/(r2**6) - ss6/(r2**3)
                  ff = 12.*epsRod*ff/r2
                  ! ==== DEBUG ====
                  if (ff > 1) then
                      ff = 1
                      print*, 'too much'
                  endif
                  ! ===============
                  velX(i) = velX(i) - ff*dx
                  velY(i) = velY(i) - ff*dy

                  velXrod = velXrod + ff*dx
                  velYrod = velYrod + ff*dy                  
                  torquerod = torquerod + ff * (drodx*dy-drody*dx)
                  !print*, 'Force: ', ff*dx, ff*dy, 'Torque: ', ff * (drodx*dy-drody*dx), 'Irod: ', Irod 
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

    ! =============================
    ! move rod degrees of freedom
    ! =============================

      rodXcm = rodXcm + dt*velXrod/Nrod/massRod
      rodYcm = rodYcm + dt*velYrod/Nrod/massRod
      rodtheta = rodtheta + dt*torquerod/Irod ! FAKE Inertia

    ! =============================
    ! transform rod 
    ! =============================

      do i = 1, Nrod
          new_XY_rod(i,1) = (i-(Nrod+1)/2.0)*cos(rodtheta)*1.0 + rodXcm
          new_XY_rod(i,2) = (i-(Nrod+1)/2.0)*sin(rodtheta)*1.0 + rodYcm
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


subroutine get_o_r_rod(X, Y, Theta, Xrod, Yrod, oldXrod, oldYrod, mode, rotDir, old_rotDir, Nobs, N, Nrod, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! ===========================================
    implicit none
    real, intent(in)    :: X(N), Y(N), Theta(N)
    real, intent(in)    :: Xrod(Nrod), Yrod(Nrod)
    real, intent(in)    :: oldXrod(Nrod), oldYrod(Nrod)
    integer, intent(in) :: N, Nrod, Nobs, mode, rotDir, old_rotDir
    real, intent(out)   :: Obs(N, Nobs), Rew(N)
    integer :: i, j, n_cone
    real :: dx, dy, r, dtheta, val, th, cmRod(2), oldcmRod(2)
    real :: dRodtheta, dRod, rotRod
    real :: a, b, ss=4.8, near, torque
    real, parameter :: PI = 3.14159265358979323846264

    Obs = 0
    Rew = 0
  
    
    cmRod(1) = SUM(Xrod)/Nrod
    cmRod(2) = SUM(Yrod)/Nrod

    oldcmRod(1) = SUM(oldXrod)/Nrod
    oldcmRod(2) = SUM(oldYrod)/Nrod

    dRod = sqrt((oldcmRod(2)-cmRod(2))**2 + (oldcmRod(1)-cmRod(1))**2 )

    dRodtheta = atan2(cmRod(2) - oldcmRod(2), cmRod(1) - oldcmRod(1))

    rotRod = atan2(   Yrod(Nrod)-   Yrod(1),   Xrod(Nrod)-   Xrod(1)) - &
             atan2(oldYrod(Nrod)-oldYrod(1),oldXrod(Nrod)-oldXrod(1))
    rotRod = rotRod / (2*PI) - floor(rotRod / (2*PI) + 0.5)

    ! =============================
    ! seeing other particles
    ! =============================

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
            val = (ss/r)
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(i,n_cone) = Obs(i,n_cone)+val
            endif
            
            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(j,n_cone) = Obs(j,n_cone)+val
            endif
        enddo
    enddo

    ! seeing the rod particles + rewards
    do i = 1, N

        a = Theta(i)  ! orientation of particle respect to x-axis.
        b = dRodtheta ! direction of motion of rod.
 
        near = 0

        do j = 1, Nrod
            dx = Xrod(j)-X(i)
            dy = Yrod(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            ! particle sees rod 
            th = (dtheta - Theta(i))/2./PI
            th = th - floor(th + 0.5)
            n_cone = floor(th*10+0.5) + 3
            !print*, X(i), Y(i), Theta(i), Xrod(j), Yrod(j), th, n_cone

            val =  (ss/r) / Nrod
            if ((n_cone < 6) .and. (n_cone>0)) then
                Obs(i,n_cone+5) = Obs(i,n_cone+5)+val
                if (r < 2.*ss) near = 1.
           endif

        enddo
        dx = cmRod(1) - X(i)
        dy = cmRod(2) - Y(i)
        r = sqrt(dx*dx + dy*dy)
        torque = cos(a)*dy - sin(a)*dx
        
        ! different reward functions to choose from
        select case (mode)
            case (1)
                Rew(i) = reward_move(r/ss, dRod, a, b, near)
                !print*, i, 'x ', X(i),'y ', Y(i), ' theta ', a, 'rodtheta ', b,&
                !       'a-b ', a-b, ' mod2pi ', ((a-b) - floor((a-b)/2.d0/PI+0.5d0)*2*PI), &
                !       reward_move(r/ss, dRod, a, b, near)
            case (2)
                Rew(i) = reward_move_back(r/ss, dRod, a, b, near)
                Obs(i, 11) = cos(a)
                Obs(i, 12) = sin(a)
            case (3) 
                ! reward positive irrespective to direction of rotation
                ! no penalty for translation of center of mass.
                Rew(i) = reward_rotate(abs(rotRod), torque, near)
            case (4) 
                ! reward positive only if clockwise (-1) or anti-clockwise (+1).
                ! with penalty for translation of center of mass.
                Rew(i) = reward_rotate(abs(rotRod), torque*old_rotDir, near)
                Obs(i, 11 + int((rotDir+1)/2)) = 1.
            case (5) ! debug reward for contact
                Rew(i) = r/ss * near
        end select
        Rew(i) = Rew(i) + sum(Obs(i,6:10))
    enddo

    return

contains

    real FUNCTION reward_rotate(rotRod, tq, near)
    ! reward function for rotation on the spot
      implicit none
      !
      real :: rotRod, tq, near
      reward_rotate =  (rotRod * tq) * near * 10.! - dRod
      return
    end function reward_rotate    

    real FUNCTION reward_move_back(rss, dRod, a, b, near)
    ! reward function for linear translation in direction (-x)
      implicit none 
      !
      real :: rss, a, b, near, dRod
      reward_move_back =  -cos(b) * dRod * cos(a)**2 / rss * near * 10.
      return
    end function reward_move_back    
    
    real FUNCTION reward_move(rss, dRod, a, b, near)
    ! Reward function for linear translation in any direction. 
    ! Maximum reward when particle is aligned with rod direction.
      implicit none
      !  real :: rand ! using old generator
      real :: rss, a, b, near, dRod, ab_half
      ab_half = ((a-b) - floor((a-b)/2./PI+0.5)*2*PI)/2.
      reward_move = dRod * cos(ab_half) / rss * near * 10.
      return
    end function reward_move
    
end subroutine
