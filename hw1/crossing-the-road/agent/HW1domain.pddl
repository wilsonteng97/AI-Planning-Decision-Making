(define (domain grid_world ) 
(:requirements :strips :typing) 
(:types car
agent - car
gridcell
time
) 
(:predicates (at ?pt1 - gridcell ?car - car) 
(at_time ?t1 - time) 
(next_time ?t1 - time ?t2 - time) 
(up_next ?pt1 - gridcell ?pt2 - gridcell) 
(down_next ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next ?pt1 - gridcell ?pt2 - gridcell) 
(blocked ?pt1 - gridcell ?t2 - time) 
) 
(:action UP
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent ?t1 - time ?t2 - time) 
:precondition (and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (up_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))
:effect (and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))
) 
(:action DOWN
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent ?t1 - time ?t2 - time) 
:precondition (and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (down_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))
:effect (and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))
) 
(:action FORWARD
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent ?t1 - time ?t2 - time) 
:precondition (and (at ?pt1 ?agt) (at_time ?t1) (next_time ?t1 ?t2) (forward_next ?pt1 ?pt2) (not (blocked ?pt2 ?t2)))
:effect (and (not (at ?pt1 ?agt)) (not (at_time ?t1)) (at ?pt2 ?agt) (at_time ?t2))
) 
) 
