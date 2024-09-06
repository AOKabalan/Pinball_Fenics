// Parameters
R = 0.5; // Radius of the cylinders
L = 3.0 * R; // Side length of the equilateral triangle

alfa = 1.2;
// Coordinates of the vertices of the equilateral triangle
x1 = -L * Sqrt(3.0) / 2.0; y1 = 0;
x2 = 0; y2 = -1.5 * R;
x3 = 0; y3 = 1.5 * R;

// Cylinder 1
Point(1) = {x1, y1, 0, 1.0};
Point(2) = {x1 + R, y1, 0, 1.0};
Point(3) = {x1 - R, y1, 0, 1.0};
Point(4) = {x1, y1 + R, 0, 1.0};
Point(5) = {x1, y1 - R, 0, 1.0};
Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 2};
Line Loop(1) = {1, 2, 3, 4};
//Plane Surface(1) = {1};

// Cylinder 2
Point(6) = {x2, y2, 0, 1.0};
Point(7) = {x2 + R, y2, 0, 1.0};
Point(8) = {x2 - R, y2, 0, 1.0};
Point(9) = {x2, y2 + R, 0, 1.0};
Point(10) = {x2, y2 - R, 0, 1.0};
Circle(5) = {7, 6, 9};
Circle(6) = {9, 6, 8};
Circle(7) = {8, 6, 10};
Circle(8) = {10, 6, 7};
Line Loop(2) = {5, 6, 7, 8};
//Plane Surface(2) = {2};

// Cylinder 3
Point(11) = {x3, y3, 0, 1.0};
Point(12) = {x3 + R, y3, 0, 1.0};
Point(13) = {x3 - R, y3, 0, 1.0};
Point(14) = {x3, y3 + R, 0, 1.0};
Point(15) = {x3, y3 - R, 0, 1.0};
Circle(9) = {12, 11, 14};
Circle(10) = {14, 11, 13};
Circle(11) = {13, 11, 15};
Circle(12) = {15, 11, 12};
Line Loop(3) = {9, 10, 11, 12};
//Plane Surface(3) = {3};

// Bounding box
Point(16) = {-6, -6, 0};
Point(17) = {20, -6, 0};
Point(18) = {20, 6, 0};
Point(19) = {-6, 6, 0};


Line(13) = {17, 18};

Line(14) = {19, 16};


// BOXES FOR MESH REFINEMENET

Point(20) = {1.5, -6, 0};
Point(21) = {1.5, 6, 0};

//Line(15) = {20,21};


Point(22) = {14, -6, 0};
Point(23) = {14, 6, 0};

//Line(16) = {22,23};
beta = 2.5;
Point(28) = {1.5,beta, 0};
Point(29) = {1.5, -beta, 0};
 
Point(30) = {14, beta, 0};
Point(31) = {14, -beta, 0};
 
//Point(24) = {L, alfa*L, 0};
//Point(25) = {L, -alfa*L, 0};

//Line(17) = {24 ,25};

//Point(26) = {-L/1.5+ x1, alfa*L, 0};
//Point(27) = {-L/1.5 + x1, -alfa*L, 0};

//Line(18) = {26,27};
//Line(19) = {24,26};
//Line(20) = {25,27};

Line(21) = {16, 20};
Line(22) = {20, 22};
Line(23) = {22, 17};
Line(24) = {18, 23};
Line(25) = {23, 21};
Line(26) = {21, 19};

Line(27) = {21, 28};
Line(28) = {29, 20};
Line(29) = {28, 29};

Line(30) = {23, 30};
Line(31) = {30, 31};
Line(32) = {31, 22};

Line(33) = {28, 30};
Line(34) = {31, 29};

Line Loop(4) = {23, 13, 24, 30, 31, 32};
Line Loop(5) = {33, -30, 25, 27};
Line Loop(6) = {-34, -31, -33, 29};
Line Loop(7) = {22, -32, 34, 28};
Line Loop(8) = {21, -28, -29, -27, 26, 14};

Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8,1,2,3};



RHO1 = 20;

Transfinite Curve{14} = RHO1*1.2;
Transfinite Curve{29,31} = RHO1*1.7;
Transfinite Curve{33,34} = RHO1*3;
Transfinite Curve{27,28} = RHO1/2;
Transfinite Curve{1,2,3,4} = RHO1*1.5;
Transfinite Curve{5,6,7,8} = RHO1*1.5;
Transfinite Curve{9,10,11,12} = RHO1*1.5;
Transfinite Curve{13} = RHO1/1.7;
Transfinite Curve{24,23} = RHO1/2;
Transfinite Curve{25,22} = RHO1*1.2;
Transfinite Curve{31} = RHO1/1.5;
//Line Loop(5) = {13, 14, 15, 18}

//Line Loop
//Boolean difference to subtract cylinders from the flow domain
//BooleanDifference{ Surface{4}; Delete; }{ Surface{1, 2, 3}; Delete; };

// Physical groups for boundary conditions
Physical Surface("FluidDomain") = {4,5,6,7,8};

Physical Curve(1) = {14}; //Inlet
Physical Curve(2) = {13};//Outlet"
Physical Curve(3) = {21, 22, 23, 24, 25, 26};//"Walls"
Physical Curve(4) = {1,2,3,4};//"Cylinder1"
Physical Curve(5) = {5,6,7,8};//"Cylinder2"
Physical Curve(6) = {9,10,11,12};//"Cylinder3"


// Mesh settings
Mesh.CharacteristicLengthMin = 0.2;
Mesh.CharacteristicLengthMax = 0.8;

// Generate the 2D mesh


Mesh 2;
Save "mid_pinball.msh";

