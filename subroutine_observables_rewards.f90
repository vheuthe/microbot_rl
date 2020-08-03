
subroutine get_o_r_mix_tasks(X, Y, Theta, Species, cost, mode, switch, obs_type, cone_angle, flag_LOS, N, NObs, Obs, Rew)
! ===========================================
! gets observables and rewards from positions
! mode 
! switch 
! X = array of X positions
! Y = array of Y positions
! Theta = array of orientations [rad]
! Species = integer array of species: -1 or +1 !!!!
! cost = reward parameter for the "cost" of seing particles from other species
! mode = indicates whether mix (mode = 1), demix (mode = 2) or switch (mode = 3)  - USE MODE 2 for demixing
! switch = determines whether to mix (1 = mixing) or demix (0 = demix)            - not used in MODE 2
! obs_type = observables function "1": 6.8/r,  "2": 6.8/r*r
! cone_angle = TOTAL angle of sight [rad]   
! flag_LOS = whether to include blocking of sight
! N = number of particles
! NObs = number of observables
! Obs, Rew are given as output when called from python
! ===========================================
    implicit none
    real , intent(in) :: X(N), Y(N), Theta(N), cost, cone_angle
    integer, intent(in) :: N, NObs, switch, mode, obs_type
    integer, intent(in) :: Species(N) 
    real , intent(out) :: Obs(N,NObs), Rew(N)
    integer :: i, j, k, other, n_cone, cones=-1
    real :: dx, dy, r, th, dtheta, val, sp_th
    real :: in_sight, covered_l, covered_r
    real :: vision_l, vision_r
    real :: dx2, dy2, r2, dtheta2, dark, ss=6.2, cone_slice
    real, allocatable :: edge(:)
    real, parameter :: PI = 3.14159265358979323846264
    logical, intent(in) :: flag_LOS

    Obs = 0
    Rew = 0
    
    ! calculate real number of sight cones
    select case (mode)
    case (1) ! pure mixing
        cones = NObs / 2
    case (2) ! pure demixing
        cones = NObs / 2
    case (3) ! switch mixing/demixing
        cones = (NObs - 2) / 2
    end select
    
    ! calcuation of edges of cone of sight
    ! for smooth vision
    allocate(edge(cones+1))
    do i = 0, cones
        edge(i+1) = -cone_angle/2. + cone_angle*i/cones  
    enddo 
    cone_slice = cone_angle / cones

    ! loop over pairs
    do i = 1, N-1
        do j = i+1, N
        
            ! other is 1 if the two species are different, 0 if they are the same.
            other = (1-(Species(i)*Species(j)) ) / 2
            
            dx = X(j)-X(i)
            dy = Y(j)-Y(i)
            r = sqrt(dx*dx + dy*dy)
            dtheta = atan2(dy,dx)
            sp_th = atan(ss, r)/2.
            ! particle j as seen by particle i
            th = (dtheta - Theta(i))/2./PI
            th = (th - floor(th + 0.5))*2*PI
            ! th is angle from particle i to j, relative to orientation.
            ! sp_th is angular "width" of partical j as seen by i.
            ! th goes from [-pi, pi]
             
            
                        
            if (obs_type == 1) then 
                val = (6.8/r)
            else if (obs_type == 2) then
                val = (6.8/r**2)
            else 
                print*, 'ERROR NO OBS_TYPE IS DEFINED!'
                STOP
            endif
            
            ! partial covering is divided in covering from left and from right.
            ! (particles behind are always 100% covered.)
            covered_l = 0
            covered_r = 0   
            
            ! if particle j total width is inside cone of sight do:
            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                ! terribly expensive way
                ! to account for line of sight

                if (flag_LOS) then
                
                    ! check on all other particles
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
                       
                        ! blocking is angular distance between centers, minus sum of widths of particles 
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
                
                ! vision_l, vision_r are visible edges of particle j from the point of view of particle i.
                vision_l = th+sp_th-covered_l
                vision_r = th-sp_th+covered_r


                do n_cone= 1, cones
                    ! in_sight is fraction of particle inside cone of sight.
                    in_sight = 0.
                    
                    in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                    
                    ! if other = 0 (same) -> n_cone is observable index
                    ! if other = 1 (different) -> n_cone + cones is observable index
                    ! n_cone is cone of sight
                    Obs(i,n_cone+other*cones) = Obs(i,n_cone+other*cones)+val*in_sight
                    
                    if (Obs(i, n_cone + other*cones) > 6.8/5.5 * (cone_slice) / atan(6.2/6.) ) then
                        print*, "ERROR"
                    endif
                    
                enddo
                
            !    Rew(i) = Rew(i)+val*(1.-other*(1+cost))
            endif
            
            ! same calculations, but for particle i as seen by particle j
            ! j to i
            th = (dtheta + PI - Theta(j))/2./PI
            th = (th - floor(th + 0.5))*2*PI
            ! th is in [-pi, pi]

            covered_l = 0
            covered_r = 0   

            ! if particle i total width is inside cone of sight of particle j do:
            if ((th>-(cone_angle/2.+sp_th)).and.(th<(cone_angle/2.+sp_th))) then
                ! terribly expensive way
                ! to account for line of sight
                if (flag_LOS) then
                    do k = 1, N 
                        if ((i==k).or.(j==k)) cycle
                        dx2 = X(k)-X(j)
                        dy2 = Y(k)-Y(j)
                        r2 = sqrt(dx2*dx2 + dy2*dy2)
                        
                        if (r2 > r) cycle !only closer particles can obscure
                        
                        dtheta2 = atan2(dy2,dx2)
                        dtheta2 = (dtheta2 - Theta(j))/2./PI
                        dtheta2 = (dtheta2 - floor(dtheta2 + 0.5))*2*PI
                        dark = atan(ss, r2)/2.
                        
                        ! blocking is angular distance between centers, minus sum of widths of particles 
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
                    ! in_sight is fraction of particle inside cone of sight.
                    in_sight = 0.
                                        
                    in_sight = max((min(vision_l, edge(n_cone+1)) - max(vision_r, edge(n_cone))), 0.) /sp_th/2. 
                    
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
