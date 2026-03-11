import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PyramidLoader } from './pyramid-loader';

describe('PyramidLoader', () => {
  let component: PyramidLoader;
  let fixture: ComponentFixture<PyramidLoader>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PyramidLoader],
    }).compileComponents();

    fixture = TestBed.createComponent(PyramidLoader);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
