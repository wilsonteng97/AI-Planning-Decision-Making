(define (domain grid_world ) 
(:requirements :strips :typing) 
(:types car
agent - car
gridcell
) 
(:predicates (at ?pt1 - gridcell ?car - car) 
(up_next ?pt1 - gridcell ?pt2 - gridcell) 
(down_next ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next ?pt1 - gridcell ?pt2 - gridcell) 
(blocked ?pt1 - gridcell) 
) 
(:action UP
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent) 
:precondition (and (at ?pt1 ?agt) (not (blocked ?pt2)) (up_next ?pt1 ?pt2))
:effect (and (not (at ?pt1 ?agt)) (at ?pt2 ?agt))
) 
(:action DOWN
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent) 
:precondition (and (at ?pt1 ?agt) (not (blocked ?pt2)) (down_next ?pt1 ?pt2))
:effect (and (not (at ?pt1 ?agt)) (at ?pt2 ?agt))
) 
(:action FORWARD
:parameters ( ?pt1 - gridcell ?pt2 - gridcell ?agt - agent) 
:precondition (and (at ?pt1 ?agt) (not (blocked ?pt2)) (forward_next ?pt1 ?pt2))
:effect (and (not (at ?pt1 ?agt)) (at ?pt2 ?agt))
) 
) 
